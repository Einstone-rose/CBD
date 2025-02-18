import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import comb

class pre_norm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class feed_forward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    
class fsp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fsp_part = nn.Sequential(
            nn.Linear(dim, int(dim / 2)),
            nn.ReLU(),
            nn.Linear(int(dim / 2), dim),
        )

    def forward(self, x): 
        # input decode tokens shape is [batch size, patch num, dim]
        x = x.transpose(-1, -2)
        freq_x = torch.fft.fft(x)
        freq_x_stack = torch.stack([freq_x.real, freq_x.imag], dim = 1).transpose(-1, -2)  # [batch size, 2, patch num, dim]
        freq_x_mask = torch.mean(freq_x_stack, dim=-2, keepdim=True)  # [batch size, 2, 1, dim]
        freq_x_mask = torch.sigmoid(self.fsp_part(freq_x_mask))
        freq_x_stack = freq_x_stack * freq_x_mask
        freq_x = torch.complex(freq_x_stack[:, 0, :, :], freq_x_stack[:, 1, :, :]).transpose(-1, -2)  # [batch size, dim, patch num]
        spatial_x = torch.fft.ifft(freq_x).abs()
        spatial_x = spatial_x.transpose(-1, -2)
        return spatial_x


class fre_mlp_fl(nn.Module):
    # frequency mlp operated on fl dimension
    def __init__(self, dim, fre_num):
        super().__init__()
        self.embed_size = dim
        self.scale = 0.02
        self.fre_num = int(fre_num/2+1) if fre_num % 2 == 0 else int((fre_num+1)/2+1)
        self.sparsity_threshold = 0.01
        self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.fc_energy = nn.Linear(self.fre_num, self.fre_num)

    def forward(self, x):
        # input x shape is [bs, fl, dm]
        bs, fl, dm = x.shape
        if fl % 2 != 0:
            x = torch.cat((x, x[:, -1:, :].expand(bs, 1, dm)), dim=1)

        x = x.transpose(-1, -2)
        x = torch.fft.rfft(x, norm='ortho').transpose(-1, -2)
        origin_ffted = x
        o1_real = torch.zeros([x.shape[0], x.shape[1], self.embed_size], device=x.device)
        o1_imag = torch.zeros([x.shape[0], x.shape[1], self.embed_size], device=x.device)
        o1_real = F.relu(
            torch.einsum('bld,de->ble', x.real, self.r1) + torch.einsum('bld,de->ble', x.imag, self.i1) + self.rb1
        )
        o1_imag = F.relu(
            torch.einsum('bld,de->ble', x.real, self.i1) - torch.einsum('bld,de->ble', x.imag, self.r1) + self.ib1
        )
        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, self.sparsity_threshold)
        y = torch.view_as_complex(y)
        y = y * origin_ffted
        # spectrum rebalance y:[bs, sq_len, dim]
        energy = torch.sqrt(torch.sum(origin_ffted.real ** 2 + origin_ffted.imag ** 2, dim=-1, keepdim=True)) # [bs, sq_len, 1]
        print(f'energy: {energy}')
        energy = self.fc_energy(energy.squeeze(-1))
        balance_weight = 1 - torch.sigmoid(energy)
        y = balance_weight.unsqueeze(-1) * y
        y = torch.fft.irfft(y.transpose(-1, -2), norm='ortho').abs().transpose(-1, -2)
        y = y[:, :fl, :]
        
        return y
    
    
class fre_mlp_dong(nn.Module):
    # frequency mlp operated on fl dimension
    def __init__(self, dim, fre_num):
        super().__init__()
        self.embed_size = dim
        self.scale = 0.02
        self.fre_num = int(fre_num/2+1) if fre_num % 2 == 0 else int((fre_num+1)/2+1)
        self.sparsity_threshold = 0.01
        self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.K = 12
        self.fc_bern = nn.Linear(self.fre_num, self.K)
        
    def forward(self, x):
        # input x shape is [bs, fl, dm]
        bs, fl, dm = x.shape
        if fl % 2 != 0:
            x = torch.cat((x, x[:, -1:, :].expand(bs, 1, dm)), dim=1)

        x = x.transpose(-1, -2)
        x = torch.fft.rfft(x, norm='ortho').transpose(-1, -2)
        # Frequency-based Global Enhancer (FGE)
        origin_ffted = x
        o1_real = torch.zeros([x.shape[0], x.shape[1], self.embed_size], device=x.device)
        o1_imag = torch.zeros([x.shape[0], x.shape[1], self.embed_size], device=x.device)
        o1_real = F.relu(
            torch.einsum('bld,de->ble', x.real, self.r1) + torch.einsum('bld,de->ble', x.imag, self.i1) + self.rb1
        )
        o1_imag = F.relu(
            torch.einsum('bld,de->ble', x.real, self.i1) - torch.einsum('bld,de->ble', x.imag, self.r1) + self.ib1
        )
        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, self.sparsity_threshold)
        y = torch.view_as_complex(y)
        y = y * origin_ffted
        # Bernstein Spectrum Balancer (BSB)
        energy_wo_norm = torch.sqrt(torch.sum((origin_ffted.real) ** 2 + (origin_ffted.imag) ** 2, dim=-1, keepdim=True))
        # min_vals, _ = energy_wo_norm.min(dim=1, keepdim=True)
        # max_vals, _ = energy_wo_norm.max(dim=1, keepdim=True)
        # energy_w_norm = (energy_wo_norm - min_vals) / (max_vals - min_vals) # [45, 9, 8, 1]
        # print(f'energy_w_norm: {energy_w_norm}')
        energy = torch.log(energy_wo_norm.squeeze(-1)) # 1 + [-1, 1] = [0, 2]
        energy = energy.masked_fill(energy == 0.0, -1e9)
        # print(f'energy: {energy}')
        energy = torch.softmax(energy, dim=-1)
        
        # convert the slope of energy to [0, 2]
        att_w = self.fc_bern(energy_wo_norm.squeeze(-1))
        att_w = torch.sigmoid(att_w) # [fre_num, K]
        # print(f'att_w: {att_w}')
        # print(f'energy: {energy}')
        out = att_w[:, 0].unsqueeze(-1) * comb(self.K, 0) * (1-energy)**self.K
        for k in range(1, self.K): # Bernstein Approximation
            out = out + att_w[:, k].unsqueeze(-1) * comb(self.K, k) * (1-energy)**(self.K-k) * (energy**k) 
        y = out.unsqueeze(-1) * y
        y = torch.fft.irfft(y.transpose(-1, -2), norm='ortho').abs().transpose(-1, -2)
        y = y[:, :fl, :]
        return y


class frequency_decoder(nn.Module):
    def __init__(self, dim, depth, mlp_dim, fl, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.ln = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                pre_norm(dim, fre_mlp_dong(dim, fre_num=fl)),
                pre_norm(dim, feed_forward(dim, mlp_dim, dropout))
            ]))
        # self.conv = nn.Conv1d(2 * dim, dim, kernel_size=1, stride=1, padding=1)

    def forward(self, x):
        # x is [bs, fl, dm]
        for fre_mlp_fl, ff in self.layers:
            x = fre_mlp_fl(x) + x
            # x = self.conv(torch.cat([x, fre_mlp_fl(x)], dim=-1).transpose(-1, -2)).transpose(-1, -2)
            x = ff(x) + x
        x = self.ln(x)  # [batch size, fl, dim]
        return x
