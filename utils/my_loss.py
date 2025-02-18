import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
if IS_HIGH_VERSION:
    import torch.fft as fft

class TimeRebuildLoss(torch.nn.Module):
    def __init__(self, device, args) -> None:
        super(TimeRebuildLoss, self).__init__()
        self.args = args
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mse = torch.nn.MSELoss(reduction='mean')

    def forward(self, origin_data, recon_data):
        # origin_data = get_normalize_value(origin_data)
        # recon_data = get_normalize_value(recon_data)
        # print(f'recon_data type is {type(recon_data)}')
        # print(f'origin_data type is {type(origin_data)}')
        # print(f'recon_data shape is {recon_data.shape}')
        # print(f'origin_data shape is {origin_data.shape}')
        ts_loss = self.mse(recon_data, origin_data)
        return ts_loss
    
class Codebook_recon:
    def __init__(self):
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.2)
    
    def compute(self, token_prediction_prob, tokens):
        recon_loss = self.ce(token_prediction_prob.view(-1, token_prediction_prob.shape[-1]), tokens.view(-1))
        return recon_loss
    
class Align:
    def __init__(self):
        self.mse = nn.MSELoss(reduction='mean')
        self.ce = nn.CrossEntropyLoss()

    def compute(self, rep_mask, rep_mask_prediction):
        align_loss = self.mse(rep_mask, rep_mask_prediction)
        return align_loss

class InfoNCELoss(torch.nn.Module):
    def __init__(self):
        super(InfoNCELoss, self).__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
    
    def forward(self, s_output, w_output, freq=False):  # [batch_size, seq_len, nvars]
        s_output = s_output.permute(0, 2, 1)  # [batch_size, nvars, seq_len]
        w_output = w_output.permute(0, 2, 1)  # [batch_size, nvars, seq_len]
        s_output = s_output.view(-1, s_output.shape[-1])  # [batch_size * nvars, seq_len]
        w_output = w_output.view(-1, w_output.shape[-1])  # [batch_size * nvars, seq_len]
        # s_output = s_output.view(s_output.shape[0], -1)  # [batch_size, nvars * seq_len]
        # w_output = w_output.view(w_output.shape[0], -1)  # [batch_size, nvars * seq_len]
        # calculate cosine similarity
        if freq:
            s_output = fft.fft(s_output, norm='ortho').abs()
            w_output = fft.fft(w_output, norm='ortho').abs()
        similarity = torch.matmul(s_output, w_output.transpose(0, 1))  # [batch_size * nvars, batch_size * nvars]
        # apply softmax to convert similarity to probability
        prob = self.softmax(similarity)  # [batch_size * nvars, batch_size * nvars]
        # calculate loss
        bs = s_output.shape[0]
        positive_samples = prob[torch.arange(bs), torch.arange(bs)]  # [batch_size * nvars]
        negtive_samples = prob.sum(dim=-1) - positive_samples  # [batch_size * nvars]
        loss = -torch.log((positive_samples / (negtive_samples + 1e-8)).clamp(min=1e-8)).mean()
        return loss

class MyFocalFrequencyLoss(nn.Module):
    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super(MyFocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x: torch.Tensor):
        # input x shape is [batch size, nvars, seq_len]
        batch_size, nvars, seq_len = x.shape
        reshape_x = x.reshape(batch_size * nvars, seq_len)  # [batch size * nvars, seq_len]
        # mean = reshape_x.mean(dim=-1, keepdim=True)
        # var = reshape_x.var(dim=-1, keepdim=True)
        # reshape_x = (reshape_x - mean) / (var + 1.e-6)**.5
        if IS_HIGH_VERSION:
            freq = fft.rfft(reshape_x, norm='ortho')
            freq = torch.stack([freq.real, freq.imag], -1)
        else:
            freq = torch.rfft(reshape_x, 1, onesided=False, normalized=True)
        return freq
    
    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # print(f'recon_freq shape is {recon_freq.shape}')  # should be [batch_size * nvars, seq_len, 2]
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha  # this is w(u,v)

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                # matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)
    
    def forward(self, pred, target, is_freq=False, matrix=None, **kwargs):
        # input x shape should be [batch size, nvars, seq_len]
        if is_freq:
            batch_size, nvars, seq_len = pred.shape
            pred = pred.reshape(batch_size * nvars, seq_len)
            pred_freq = torch.stack([pred.real, pred.imag], -1)
        else:
            pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)
        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)
        # calculate focal frequency loss
        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight

