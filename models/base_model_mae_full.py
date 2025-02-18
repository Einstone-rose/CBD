from torch import nn
import torch
import numpy as np
import random
from layer.frequency_decoder import frequency_decoder as frequency_decoder

class BaseModel(nn.Module):
    def __init__(self, configs, task_name):
        super(BaseModel, self).__init__()

        self.configs = configs
        self.seq_len = configs.seq_len
        self.task_name = task_name
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            # nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.patch_to_embedding = nn.Linear(configs.final_out_channels, configs.TC.hidden_dim)
        self.encode_layer = nn.TransformerEncoderLayer(d_model=configs.TC.hidden_dim, nhead=configs.encode_head_num, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encode_layer, num_layers=configs.encoder_layer_num)
        self.mask_token = nn.Parameter(torch.randn(configs.TC.hidden_dim,))
        self.mask_len = int(configs.features_len * configs.mask_ratio)

        self.decode_layer_spa = nn.TransformerEncoderLayer(d_model=configs.TC.hidden_dim, nhead=configs.decode_head_num, batch_first=True)
        self.trans_decoder_spa = nn.TransformerEncoder(encoder_layer=self.decode_layer_spa, num_layers=configs.decode_layer_num)
        self.frequency_decoder = frequency_decoder(dim=configs.TC.hidden_dim, depth=configs.freq_decode_layer, mlp_dim=configs.TC.hidden_dim//2, fl=configs.features_len)
        self.embedding_to_patch_spa = nn.Linear(configs.TC.hidden_dim, configs.final_out_channels)
        self.embedding_to_patch_freq = nn.Linear(configs.TC.hidden_dim, configs.final_out_channels)

        self.deconv_block1 = nn.Sequential(
            nn.ConvTranspose1d(32, configs.input_channels, kernel_size=configs.kernel_size,
                       stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.input_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
        )

        self.deconv_block_freq1 = nn.Sequential(
            nn.ConvTranspose1d(32, configs.input_channels, kernel_size=configs.kernel_size,
                       stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.input_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
        )

        self.deconv_block2 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )

        self.deconv_block_freq2 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )

        self.deconv_block3 = nn.Sequential(
            nn.ConvTranspose1d(configs.final_out_channels, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )

        self.deconv_block_freq3 = nn.Sequential(
            nn.ConvTranspose1d(configs.final_out_channels, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )

        model_output_dim = configs.features_len
        self.dropout = nn.Dropout(p=0.1)
        self.logits = nn.Sequential(
            nn.Linear(model_output_dim * configs.TC.hidden_dim, model_output_dim * configs.TC.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(model_output_dim * configs.TC.hidden_dim // 2, configs.num_classes)
        )

    def pretrain(self, input_data: torch.Tensor):
        # input data shape is (batch_size, channels, seq_len)
        input_data = input_data.transpose(1, 2)
        # normalization
        # for har,awr,sad,ecg,fb and uwave, train without normalization could be better
        # cause their range itself is not large
        means = input_data.mean(1, keepdim=True).detach()
        input_data = input_data - means
        stdev = torch.sqrt(torch.var(input_data, dim=1, keepdim=True, unbiased=False) + 1e-5)
        input_data /= stdev

        # conv block
        x_in = input_data.transpose(1, 2)
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)  # [batch_size, final_out_channels, feature_len]

        # mask module
        enc_x = x.transpose(1, 2)
        enc_x = self.patch_to_embedding(enc_x)  # convert channels to embedding dim
        rep_mask_token = self.mask_token.repeat(enc_x.shape[0], enc_x.shape[1], 1)  # random init mask token
        index = np.arange(enc_x.shape[1])  # all feature length index
        random.shuffle(index)
        v_index = index[:-self.mask_len]  # random pick visble time step
        m_index = index[-self.mask_len:]  # random pick mask time step
        v_index = sorted(v_index)
        m_index = sorted(m_index)
        visble = enc_x[:, v_index, :]
        rep_mask_token = rep_mask_token[:, m_index, :]

        # transformer module
        enc_x = self.encoder(visble)
        index_cat = np.concatenate([v_index, m_index])
        sorted_index = sorted(range(len(index_cat)), key=lambda k: index_cat[k])
        rebuild_enc_x = torch.cat([enc_x, rep_mask_token], dim=1)  # cat at feature length dim
        rebuild_enc_x = rebuild_enc_x[:, sorted_index, :]  # sort back to original order, shape is (batch_size, feature_len, hidden_dim)

        # decode transformer module
        rebuild_enc_x_1 = self.trans_decoder_spa(rebuild_enc_x)
        rebuild_enc_x_1 = self.embedding_to_patch_spa(rebuild_enc_x_1)

        # decode spatial feature
        dec_x = rebuild_enc_x_1.transpose(1, 2)  # [batch_size, final_out_channels, features_len]
        dec_x = self.deconv_block3(dec_x)
        dec_x = self.deconv_block2(dec_x)
        dec_x = self.deconv_block1(dec_x)
        dec_x = dec_x[:, :, :self.seq_len]
        # de-normalization
        dec_x = dec_x.transpose(1, 2)  # [batch_size, seq_len, nvars]
        dec_x = dec_x * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_x = dec_x + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))

        # decode frequency feature
        rebuild_enc_x_2 = rebuild_enc_x  # [bs, fl, dm]
        freq_x = self.frequency_decoder(rebuild_enc_x_2)
        freq_x = self.embedding_to_patch_freq(freq_x)  # [bs, fl, oc]
        freq_x = self.deconv_block_freq3(freq_x.transpose(-1, -2))
        freq_x = self.deconv_block_freq2(freq_x)
        freq_x = self.deconv_block_freq1(freq_x)
        freq_x = freq_x[:, :, :self.seq_len]
        freq_x = torch.fft.rfft(freq_x, norm='ortho')
        return dec_x, freq_x

    def finetune(self, input_data: torch.Tensor):
        # input data shape is (batch_size, channels, seq_len)
        input_data = input_data.transpose(1, 2)
        
        # normalization
        means = input_data.mean(1, keepdim=True).detach()
        input_data = input_data - means
        stdev = torch.sqrt(torch.var(input_data, dim=1, keepdim=True, unbiased=False) + 1e-5)
        input_data /= stdev

        # conv block
        x_in = input_data.transpose(1, 2)
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)  # [batch_size, final_out_channels, feature_len]

        # transformer module
        enc_x = x.transpose(1, 2)
        enc_x = self.patch_to_embedding(enc_x)
        enc_x = self.encoder(enc_x)

        # classification module
        x_flat = enc_x.reshape(enc_x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits, enc_x

    def forward(self, input_data: torch.Tensor):
        if self.task_name == 'pretrain':
            return self.pretrain(input_data)
        elif self.task_name == 'finetune':
            return self.finetune(input_data)
