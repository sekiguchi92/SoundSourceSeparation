#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import sys, os
from chainer import cuda
from progressbar import progressbar
import librosa
import soundfile as sf
import pickle as pic

from FCA import FCA
from configure import *


class MNMF(FCA):
    """ Blind Source Separation Using Multichannel Nonnegative Matrix Factorization (MNMF)

    X_FTM: the observed complex spectrogram
    covarianceMatrix_NFMM: spatial covariance matrices (SCMs) for each source
    W_NFK: basis vectors for each source
    H_NKT: activations for each source
    lambda_NFT: power spectral densities of each source (W_NFK @ H_NKT)
    """

    def __init__(self, n_source=2, n_basis=2, xp=np, init_SCM="unit", mode_update_parameter=["all", "one_by_one"][0], seed=0):
        """ initialize MNMF

        Parameters:
        -----------
            n_source: int
                the number of sources
            n_basis: int
                the number of bases of each source
            xp : numpy or cupy
            init_SCM: str
                how to initialize covariance matrix {unit, obs, ILRMA}
            mode_update_parameter: str
                'all' : update all the parameters simultanesouly to reduce computational cost
                'one_by_one' : update the parameters one by one to monotonically increase log-likelihood
        """
        super(MNMF, self).__init__(n_source=n_source, xp=xp, init_SCM=init_SCM, mode_update_parameter=mode_update_parameter, seed=seed)
        self.method_name = "MNMF"
        self.n_basis = n_basis


    def initialize_PSD(self):
        self.W_NFK = self.xp.random.rand(self.n_source, self.n_freq, self.n_basis).astype(self.xp.float)
        self.H_NKT = self.xp.random.rand(self.n_source, self.n_basis, self.n_time).astype(self.xp.float)
        self.lambda_NFT = self.W_NFK @ self.H_NKT


    def make_filename_suffix(self):
        self.filename_suffix = f"S={self.n_source}-it={self.n_iteration}-K={self.n_basis}-init={self.init_SCM}-update={self.mode_update_parameter}"

        if hasattr(self, "file_id"):
            self.filename_suffix += "-ID={}".format(self.file_id)
        print("param:", self.filename_suffix)


    def update(self):
        if self.mode_update_parameter == "one_by_one":
            self.update_axiliary_variable()
            self.update_W()
            self.update_axiliary_variable()
            self.update_H()
            self.update_axiliary_variable()
            self.update_covarianceMatrix()
        if self.mode_update_parameter == "all":
            self.update_axiliary_variable()
            self.update_WH()
            self.update_covarianceMatrix()
        self.normalize()


    def update_WH(self):
        a_1 = (self.H_NKT.transpose(0, 2, 1)[:, None] * self.tr_Cov_Yinv_X_Yinv_NFT[:, :, :, None]).sum(axis=2) # Nn F K
        b_1 = (self.H_NKT.transpose(0, 2, 1)[:, None] * self.tr_Cov_Yinv_NFT[:, :, :, None]).sum(axis=2) # Nn F K

        a_2 = (self.W_NFK[..., None] * self.tr_Cov_Yinv_X_Yinv_NFT[:, :, None]).sum(axis=1) # Nn K T
        b_2 = (self.W_NFK[..., None] * self.tr_Cov_Yinv_NFT[:, :, None]).sum(axis=1) # Nn K T

        self.W_NFK *= self.xp.sqrt(a_1 / b_1)
        self.H_NKT *= self.xp.sqrt(a_2 / b_2)


    def update_W(self):
        a_1 = (self.H_NKT.transpose(0, 2, 1)[:, None] * self.tr_Cov_Yinv_X_Yinv_NFT[:, :, :, None]).sum(axis=2) # Nn F K
        b_1 = (self.H_NKT.transpose(0, 2, 1)[:, None] * self.tr_Cov_Yinv_NFT[:, :, :, None]).sum(axis=2) # Nn F K
        self.W_NFK *= self.xp.sqrt(a_1 / b_1)
        self.lambda_NFT = self.W_NFK @ self.H_NKT + EPS


    def update_H(self):
        a_1 = (self.W_NFK[..., None] * self.tr_Cov_Yinv_X_Yinv_NFT[:, :, None]).sum(axis=1) # Nn K T
        b_1 = (self.W_NFK[..., None] * self.tr_Cov_Yinv_NFT[:, :, None]).sum(axis=1) # Nn K T
        self.H_NKT *= self.xp.sqrt(a_1 / b_1)
        self.lambda_NFT = self.W_NFK @ self.H_NKT + EPS


    def normalize(self):
        mu_NF = self.xp.trace(self.covarianceMatrix_NFMM, axis1=2, axis2=3).real
        self.covarianceMatrix_NFMM /= mu_NF[:, :, None, None]
        self.W_NFK *= mu_NF[:, :, None]

        nu_NK = self.W_NFK.sum(axis=1)
        self.W_NFK /= nu_NK[:, None]
        self.H_NKT *= nu_NK[:, :, None]

        self.lambda_NFT = self.W_NFK @ self.H_NKT + EPS


    def save_parameter(self, filename):
        param_list = [self.covarianceMatrix_NFMM, self.lambda_NFT, self.W_NFK, self.H_NKT]

        if self.xp != np:
            param_list = [cuda.to_cpu(param) for param in param_list]
        pic.dump(param_list, open(filename, "wb"))


    def load_parameter(self, filename):
        param_list = pic.load(open(filename, "rb"))
        if self.xp != np:
            param_list = [cuda.to_gpu(param) for param in param_list]

        self.covarianceMatrix_NFMM, self.lambda_NFT, self.W_NFK, self.H_NKT = param_list



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(         'input_fileName', type= str, help='filename of the multichannel observed signals')
    parser.add_argument(              '--file_id', type= str, default="None", help='file id')
    parser.add_argument(                  '--gpu', type= int, default=     0, help='GPU ID')
    parser.add_argument(             '--init_SCM', type= str, default= "obs", help='unit, obs, ILRMA')
    parser.add_argument(                '--n_fft', type= int, default=  1024, help='number of frequencies')
    parser.add_argument(                '--n_mic', type= int, default=     8, help='number of microphones')
    parser.add_argument(              '--n_basis', type= int, default=    16, help='number of basis of NMF')
    parser.add_argument(             '--n_source', type= int, default=     2, help='number of noise')
    parser.add_argument(          '--n_iteration', type= int, default=    30, help='number of iteration')
    parser.add_argument('--mode_update_parameter', type= str, default= "all", help='all, one_by_one')
    args = parser.parse_args()

    if args.gpu < 0:
        import numpy as xp
    else:
        import cupy as xp
        print("Use GPU " + str(args.gpu))
        cuda.get_device_from_id(args.gpu).use()

    wav, fs = sf.read(args.input_fileName)
    wav = wav.T
    M = min(args.n_mic, len(wav))
    for m in range(M):
        tmp = librosa.core.stft(wav[m], n_fft=args.n_fft, hop_length=int(args.n_fft/4))
        if m == 0:
            spec = np.zeros([tmp.shape[0], tmp.shape[1], M], dtype=np.complex)
        spec[:, :, m] = tmp

    separater = MNMF(n_source=args.n_source, n_basis=args.n_basis, xp=xp, init_SCM=args.init_SCM, mode_update_parameter=args.mode_update_parameter, seed=0)
    separater.load_spectrogram(spec)
    separater.file_id = args.file_id
    separater.solve(n_iteration=args.n_iteration, save_likelihood=False, save_parameter=False, save_wav=True, save_path="./", interval_save_parameter=100)
