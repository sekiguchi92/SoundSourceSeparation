#! /usr/bin/env python3
# coding:utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1d_BatchNorm_GLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', use_dropout=False, p_dropbout=0.5):
        super(Conv1d_BatchNorm_GLU, self).__init__()

        self.use_dropout = use_dropout
        self.conv1d_bn = nn.Sequential( 
            nn.Conv1d(in_channels, 2 * out_channels, kernel_size, stride, padding,
                dilation, groups, bias, padding_mode),
            nn.BatchNorm1d(2 * out_channels)
        )
        if self.use_dropout:
            self.dropout = nn.Dropout(p_dropbout)

    def forward(self, x):
        if self.use_dropout:
            return F.glu(self.dropout(self.conv1d_bn(x)), dim=1)
        else:
            return F.glu(self.conv1d_bn(x), dim=1)


class Deconv1d_BatchNorm_GLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode='zeros', use_dropout=False, p_dropbout=0.5):
        super(Deconv1d_BatchNorm_GLU, self).__init__()

        self.use_dropout = use_dropout
        self.deconv_bn = nn.Sequential(
            nn.ConvTranspose1d(in_channels, 2 * out_channels, kernel_size, stride, padding,
                output_padding, groups, bias, dilation, padding_mode),
            nn.BatchNorm1d(2 * out_channels)
        )
        if self.use_dropout:
            self.dropout = nn.Dropout(p_dropbout)

    def forward(self, x):
        if self.use_dropout:
            return F.glu(self.dropout(self.deconv_bn(x)), dim=1)
        else:
            return F.glu(self.deconv_bn(x), dim=1)


class VAE(nn.Module):

    def __init__(self, n_freq=513, n_latent=16, use_dropout=False, p_dropbout=0.5):
        super(VAE, self).__init__()
        self.n_freq = n_freq
        n_fft = (n_freq - 1) * 2
        self.n_latent = n_latent
        self.use_dropout = use_dropout
        self.p_dropbout = p_dropbout

        self.encoder = nn.Sequential(
            Conv1d_BatchNorm_GLU(n_fft // 2 + 1, n_fft // 4, kernel_size=5, stride=1, padding=2, use_dropout=use_dropout, p_dropbout=p_dropbout),
            Conv1d_BatchNorm_GLU(n_fft // 4, n_fft // 8, kernel_size=4, stride=2, padding=1, use_dropout=use_dropout, p_dropbout=p_dropbout)
        )
        self.encoder_mu = nn.Conv1d(n_fft // 8, self.n_latent, kernel_size=4, stride=2, padding=1)
        self.encoder_logvar = nn.Conv1d(n_fft // 8, self.n_latent, kernel_size=4, stride=2, padding=1)

        self.decoder = nn.Sequential(
            Deconv1d_BatchNorm_GLU(self.n_latent, n_fft // 8, kernel_size=4, stride=2, padding=1, use_dropout=use_dropout, p_dropbout=p_dropbout),
            Deconv1d_BatchNorm_GLU(n_fft // 8, n_fft // 4,  kernel_size=4, stride=2, padding=1, use_dropout=use_dropout, p_dropbout=p_dropbout),
            nn.ConvTranspose1d(n_fft // 4, n_fft // 2 + 1, kernel_size=5, stride=1, padding=2)
        )
        self.network_name = "VAE_conv1d"
        self.make_filename_suffix()


    def make_filename_suffix(self):
        self.filename_suffix = f"F={self.n_freq}-D={self.n_latent}"
        if self.use_dropout:
            self.filename_suffix += f"-DO=True-p={self.p_dropbout}"
        
        if hasattr(self, "version"):
            self.filename_suffix += f"-version={self.version}"


    def encode(self, log_x_BxFxT):
        h = self.encoder(log_x_BxFxT)
        mu_BxDxT, logvar_BxDxT = self.encoder_mu(h), self.encoder_logvar(h)

        return mu_BxDxT, logvar_BxDxT


    def decode(self, z_BxDxT):
        log_sigma_BxFxT = self.decoder(z_BxDxT)

        return log_sigma_BxFxT + 1e-6


    def forward(self, log_x_BxFxT):
        mu_BxDxT, logvar_BxDxT = self.encode(log_x_BxFxT)

        eps_BxDxT = torch.randn_like(mu_BxDxT)
        z_BxDxT = mu_BxDxT + eps_BxDxT * torch.exp(logvar_BxDxT * 0.5)

        return self.decode(z_BxDxT), mu_BxDxT, logvar_BxDxT


    def loss(self, log_x_BxFxT, length_list_T):
        batch_size = len(log_x_BxFxT)
        mask_BxT = np.zeros([batch_size, length_list_T.max()], dtype=np.int8)
        mask_BxT4 = np.zeros([batch_size, length_list_T.max()//4], dtype=np.int8)
        for b in range(batch_size):
            mask_BxT[b, :length_list_T[b]] = 1
            mask_BxT4[b, :length_list_T[b]//4] = 1
        mask_BxT = torch.as_tensor(mask_BxT).to(log_x_BxFxT.device)
        mask_BxT4 = torch.as_tensor(mask_BxT4).to(log_x_BxFxT.device)

        log_sigma_BxFxT, mu_BxDxT, logvar_BxDxT = self.forward(log_x_BxFxT)
        loss_likelihood = ((torch.exp(log_x_BxFxT - log_sigma_BxFxT) + log_sigma_BxFxT).sum(axis=1) * mask_BxT).sum() / length_list_T.sum()
        loss_KL = 0.5 * (((mu_BxDxT ** 2) + torch.exp(logvar_BxDxT) - logvar_BxDxT).sum(axis=1) * mask_BxT4).sum() / length_list_T.sum()

        return loss_likelihood + loss_KL, loss_likelihood, loss_KL


    def decode_(self, z):
        if z.ndim == 2: # D x T
            return torch.exp(self.decode(z.unsqueeze(0)).squeeze(0))
        elif z.ndim == 3: # z = N x D x T
            return torch.exp(self.decode(z))


    def encode_(self, x):
        if x.ndim == 2:
            return self.encode(torch.log(x).unsqueeze(0))[0].squeeze(0)
        else:
            return self.encode(torch.log(x))[0]


