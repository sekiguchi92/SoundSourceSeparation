#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import chainer
import sys, os
import librosa
import soundfile as sf
import time
import pickle as pic

from configure_FastModel import *
from FastFCA import FastFCA

try:
    from chainer import cuda
    FLAG_GPU_Available = True
except:
    print("---Warning--- You cannot use GPU acceleration because chainer or cupy is not installed")


class FastMNMF(FastFCA):
    """ Blind Source Separation Using Fast Multichannel Nonnegative Matrix Factorization (FastMNMF)

    X_FTM: the observed complex spectrogram
    Q_FMM: diagonalizer that converts a spatial covariance matrix (SCM) to a diagonal matrix
    G_NFM: diagonal elements of the diagonalized SCMs
    W_NFK: basis vectors for each source
    H_NKT: activations for each source
    lambda_NFT: power spectral densities of each source (W_NFK @ H_NKT)
    Qx_power_FTM: power spectra of Qx
    Y_FTM: \sum_n lambda_NFT G_NM
    """

    def __init__(self, n_source=2, n_basis=8, xp=np, init_SCM="unit"):
        """ initialize FastMNMF

        Parameters:
        -----------
            n_source: int
                the number of sources
            n_iteration: int
                the number of iteration to update all variables
            n_basis: int
                the number of bases of each source
            init_SCM: str
                how to initialize covariance matrix {unit, obs, ILRMA}
        """
        super(FastMNMF, self).__init__(n_source=n_source, xp=xp, init_SCM=init_SCM)
        self.n_basis = n_basis
        self.method_name = "FastMNMF"


    def set_parameter(self, n_source=None, n_basis=None, init_SCM=None):
        """ set parameters

        Parameters:
        -----------
            n_source: int
            n_iteration: int
            n_basis: int
            init_SCM: str
                how to initialize covariance matrix {unit, obs, ILRMA}
        """
        super(FastMNMF, self).set_parameter(n_source=n_source, init_SCM=init_SCM)
        if n_basis != None:
            self.n_basis = n_basis


    def initialize_PSD(self):
        power_observation_FT = (self.xp.abs(self.X_FTM).astype(self.xp.float32) ** 2).mean(axis=2)
        shape = 2
        self.W_NFK = self.xp.random.dirichlet(np.ones(self.n_freq)*shape, size=[self.n_source, self.n_basis]).transpose(0, 2, 1)
        self.H_NKT = self.xp.random.gamma(shape, (power_observation_FT.mean() * self.n_freq * self.n_mic / (self.n_source * self.n_basis)) / shape, size=[self.n_source, self.n_basis, self.n_time])
        self.H_NKT[self.H_NKT < EPS] = EPS
        self.lambda_NFT = self.W_NFK @ self.H_NKT


    def make_fileName_suffix(self):
        self.fileName_suffix = "S={}-it={}-L={}-init={}".format(self.n_source, self.n_iteration, self.n_basis, self.init_SCM)

        if hasattr(self, "file_id"):
            self.fileName_suffix += "-ID={}".format(self.file_id)
        else:
            print("====================\n\nWarning: Please set self.file_id\n\n====================")

        print("fileName_suffix:", self.fileName_suffix)


    def update(self):
        self.update_WH()
        self.update_CovarianceDiagElement()
        self.udpate_Diagonalizer()
        self.normalize()


    def update_WH(self):
        tmp1_NFT = (self.covarianceDiag_NFM[:, :, None] * (self.Qx_power_FTM / (self.Y_FTM ** 2))[None]).sum(axis=3)
        tmp2_NFT = (self.covarianceDiag_NFM[:, :, None] / self.Y_FTM[None]).sum(axis=3)
        a_W = (self.H_NKT[:, None] * tmp1_NFT[:, :, None]).sum(axis=3)  # N F K T M
        b_W = (self.H_NKT[:, None] * tmp2_NFT[:, :, None]).sum(axis=3)
        a_H = (self.W_NFK[..., None] * tmp1_NFT[:, :, None] ).sum(axis=1) # N F K T M
        b_H = (self.W_NFK[..., None] * tmp2_NFT[:, :, None]).sum(axis=1) # N F K T M
        self.W_NFK = self.W_NFK * self.xp.sqrt(a_W / b_W)
        self.H_NKT = self.H_NKT * self.xp.sqrt(a_H / b_H)

        self.lambda_NFT = self.W_NFK @ self.H_NKT + EPS
        self.Y_FTM = (self.lambda_NFT[..., None] * self.covarianceDiag_NFM[:, :, None]).sum(axis=0)


    def normalize(self):
        phi_F = self.xp.sum(self.diagonalizer_FMM * self.diagonalizer_FMM.conj(), axis=(1, 2)).real / self.n_mic
        self.diagonalizer_FMM = self.diagonalizer_FMM / self.xp.sqrt(phi_F)[:, None, None]
        self.covarianceDiag_NFM = self.covarianceDiag_NFM / phi_F[None, :, None]

        mu_NF = (self.covarianceDiag_NFM).sum(axis=2).real
        self.covarianceDiag_NFM = self.covarianceDiag_NFM / mu_NF[:, :, None]
        self.W_NFK = self.W_NFK * mu_NF[:, :, None]

        nu_NK = self.W_NFK.sum(axis=1)
        self.W_NFK = self.W_NFK / nu_NK[:, None]
        self.H_NKT = self.H_NKT * nu_NK[:, :, None]
        self.lambda_NFT = self.W_NFK @ self.H_NKT + EPS

        self.reset_variable()


    def save_parameter(self, fileName):
        param_list = [self.lambda_NFT, self.covarianceDiag_NFM, self.diagonalizer_FMM, self.W_NFK, self.H_NKT]
        if self.xp != np:
            param_list = [self.convert_to_NumpyArray(param) for param in param_list]

        pic.dump(param_list, open(fileName, "wb"))


    def load_parameter(self, fileName):
        param_list = pic.load(open(fileName, "rb"))
        if self.xp != np:
            param_list = [cuda.to_gpu(param) for param in param_list]

        self.lambda_NFT, self.covarianceDiag_NFM, self.diagonalizer_FMM, self.W_NFK, self.H_NKT = param_list



if __name__ == "__main__":
    import argparse
    import pickle as pic
    import sys, os

    parser = argparse.ArgumentParser()
    parser.add_argument(    'input_fileName', type= str, help='filename of the multichannel observed signals')
    parser.add_argument(         '--file_id', type= str, default="None", help='file id')
    parser.add_argument(             '--gpu', type= int, default=    0, help='GPU ID')
    parser.add_argument(           '--n_fft', type= int, default= 1024, help='number of frequencies')
    parser.add_argument(      '--n_source', type= int, default=    2, help='number of noise')
    parser.add_argument(   '--n_iteration', type= int, default=  100, help='number of iteration')
    parser.add_argument(       '--n_basis', type= int, default=    8, help='number of basis')
    parser.add_argument( '--init_SCM', type=  str, default="unit", help='unit, obs, ILRMA')
    args = parser.parse_args()

    if args.gpu < 0:
        import numpy as xp
    else:
        import cupy as xp
        print("Use GPU " + str(args.gpu))
        cuda.get_device_from_id(args.gpu).use()

    wav, fs = sf.read(args.input_fileName)
    wav = wav.T
    # M = len(wav)
    M = 3
    for m in range(M):
        tmp = librosa.core.stft(wav[m], n_fft=args.n_fft, hop_length=int(args.n_fft/4))
        if m == 0:
            spec = np.zeros([tmp.shape[0], tmp.shape[1], M], dtype=np.complex)
        spec[:, :, m] = tmp

    separater = FastMNMF(n_source=args.n_source, n_basis=args.n_basis, xp=xp, init_SCM=args.init_SCM)
    separater.load_spectrogram(spec)
    separater.file_id = args.file_id
    separater.solve(n_iteration=args.n_iteration, save_likelihood=False, save_parameter=False, save_wav=False, save_path="./", interval_save_parameter=25)
    print(separater.covarianceDiag_NFM)