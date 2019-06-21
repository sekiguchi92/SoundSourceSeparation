#!/usr/bin/env python3

import chainer
from chainer import functions as chf
from chainer import links as chl
from chainer.functions.loss.vae import gaussian_kl_divergence

from configure_VAE import *

class VAE(chainer.Chain):
    def __init__(self, n_freq=int(N_FFT/2+1), n_latent=N_LATENT, n_hidden=128):
        super(VAE, self).__init__()
        self.n_latent = n_latent

        with self.init_scope():
        # encoder
            self.linear_enc = chl.Linear(n_freq, n_hidden)
            self.linear_enc_mu  = chl.Linear(n_hidden, n_latent)
            self.linear_enc_logVar = chl.Linear(n_hidden, n_latent)

        # decoder
            self.linear_dec = chl.Linear(n_latent, n_hidden)
            self.bn_dec = chl.BatchNormalization(n_hidden)
            self.linear_dec_output = chl.Linear(n_hidden, n_freq)


    def encode(self, x):
        hidden = chf.tanh(self.linear_enc(x))
        return self.linear_enc_mu(hidden), self.linear_enc_logVar(hidden)


    def decode(self, z):
        hidden = chf.tanh(self.bn_dec(self.linear_dec(z)))
        return self.linear_dec_output(hidden)


    def encode_cupy(self, x, sampling=False):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            x_ = (chainer.Variable(x.T))
            mu, log_var = self.encode(x_)
            if sampling:
                z = chf.gaussian(mu, log_var)
                return z.data.T
            else:
                return mu.data.T


    def decode_cupy(self, z):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            z = chainer.Variable(z.T)
            x = chf.exp(self.decode(z)).data.T # exp(log(power)) = power
        return x


    def get_loss_func(self, eps=1e-8):
        def lf(x):
            mu, log_var = self.encode(x)
            batch_size = len(mu.data)

            self.vae_loss = gaussian_kl_divergence(mu, log_var) / batch_size

            z = chf.gaussian(mu, log_var)
            output_dec = chf.exp(self.decode(z)) # exp(log(power)) = power
            self.dec_loss = chf.sum(chf.log(output_dec) + x / output_dec) / batch_size

            self.loss = self.vae_loss + self.dec_loss

            return self.loss
        return lf
