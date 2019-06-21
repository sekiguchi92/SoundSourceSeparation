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

    def __init__(self, NUM_source=2, NUM_basis=8, xp=np, MODE_initialize_covarianceMatrix="unit"):
        """ initialize FastMNMF

        Parameters:
        -----------
            NUM_source: int
                the number of sources
            NUM_iteration: int
                the number of iteration to update all variables
            NUM_basis: int
                the number of bases of each source
            MODE_initialize_covarianceMatrix: str
                how to initialize covariance matrix {unit, obs, cGMM, cGMM2(only speech)}
        """
        super(FastMNMF, self).__init__(NUM_source=NUM_source, xp=xp, MODE_initialize_covarianceMatrix=MODE_initialize_covarianceMatrix)
        self.NUM_basis = NUM_basis
        self.method_name = "FastMNMF"


    def load_spectrogram(self, X_FTM):
        super(FastMNMF, self).load_spectrogram(X_FTM)
        self.W_NFK = self.xp.abs(self.xp.random.rand(self.NUM_source, self.NUM_freq, self.NUM_basis).astype(self.xp.float))
        self.H_NKT = self.xp.abs(self.xp.random.rand(self.NUM_source, self.NUM_basis, self.NUM_time).astype(self.xp.float))
        self.lambda_NFT = self.W_NFK @ self.H_NKT


    def set_parameter(self, NUM_source=None, NUM_basis=None, MODE_initialize_covarianceMatrix=None):
        """ set parameters

        Parameters:
        -----------
            NUM_source: int
            NUM_iteration: int
            NUM_basis: int
            MODE_initialize_covarianceMatrix: str
                how to initialize covariance matrix {unit, obs, cGMM}
        """
        super(FastMNMF, self).set_parameter(NUM_source=NUM_source, MODE_initialize_covarianceMatrix=MODE_initialize_covarianceMatrix)
        if NUM_basis != None:
            self.NUM_basis = NUM_basis


    def initialize_PSD(self):
        power_observation_FT = (self.xp.abs(self.X_FTM).astype(self.xp.float32) ** 2).mean(axis=2)
        shape = 2
        self.W_NFK = self.xp.random.dirichlet(np.ones(self.NUM_freq)*shape, size=[self.NUM_source, self.NUM_basis]).transpose(0, 2, 1)

        self.H_NKT = self.xp.random.gamma(shape, (power_observation_FT.mean() * self.NUM_freq * self.NUM_mic / (self.NUM_source * self.NUM_basis)) / shape, size=[self.NUM_source, self.NUM_basis, self.NUM_time])
        self.H_NKT[self.H_NKT < EPS] = EPS

        self.lambda_NFT = self.W_NFK @ self.H_NKT
        self.reset_variable()


    def make_fileName_suffix(self):
        self.fileName_suffix = "S={}-it={}-L={}-init={}".format(self.NUM_source, self.NUM_iteration, self.NUM_basis, self.MODE_initialize_covarianceMatrix)

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
        if self.xp == np:
            for f in range(self.NUM_freq):
                a_1 = self.xp.zeros([self.NUM_source, self.NUM_basis])
                b_1 = self.xp.zeros([self.NUM_source, self.NUM_basis])
                for m in range(self.NUM_mic):
                    a_1 += (self.H_NKT * (self.covarianceDiag_NFM[:, f, None, m] * (self.Qx_power_FTM[f, :, m] / (self.Y_FTM[f, :, m] ** 2))[None])[:, None]).sum(axis=2)  # N K T
                    b_1 += (self.H_NKT * (self.covarianceDiag_NFM[:, f, None, m] / self.Y_FTM[None, f, :, m])[:, None]).sum(axis=2)
                self.W_NFK[:, f] = self.W_NFK[:, f] * self.xp.sqrt(a_1 / b_1)

            self.lambda_NFT = self.W_NFK @ self.H_NKT
            self.Y_FTM = (self.lambda_NFT[..., None] * self.covarianceDiag_NFM[:, :, None]).sum(axis=0)

            for t in range(self.NUM_time):
                a_1 = self.xp.zeros([self.NUM_source, self.NUM_basis])
                b_1 = self.xp.zeros([self.NUM_source, self.NUM_basis])
                for m in range(self.NUM_mic):
                    a_1 += (self.W_NFK * (self.covarianceDiag_NFM[:, :, m] * (self.Qx_power_FTM[:, t, m] / (self.Y_FTM[:, t, m] ** 2))[None])[:, :, None] ).sum(axis=1) # N F K
                    b_1 += (self.W_NFK * (self.covarianceDiag_NFM[:, :, m] / self.Y_FTM[None, :, t, m])[:, :, None]).sum(axis=1) # N F K
                self.H_NKT[:, :, t] = self.H_NKT[:, :, t] * self.xp.sqrt(a_1 / b_1)

            self.lambda_NFT = self.W_NFK @ self.H_NKT
            self.Y_FTM = (self.lambda_NFT[..., None] * self.covarianceDiag_NFM[:, :, None]).sum(axis=0)

        else:
            a_1 = (self.H_NKT[:, None, :, :, None] * (self.covarianceDiag_NFM[:, :, None] * (self.Qx_power_FTM / (self.Y_FTM ** 2))[None])[:, :, None]).sum(axis=4).sum(axis=3)  # N F K T M
            b_1 = (self.H_NKT[:, None, :, :, None] * (self.covarianceDiag_NFM[:, :, None] / self.Y_FTM[None])[:, :, None]).sum(axis=4).sum(axis=3)
            self.W_NFK = self.W_NFK * self.xp.sqrt(a_1 / b_1)
            self.lambda_NFT = self.W_NFK @ self.H_NKT
            self.Y_FTM = (self.lambda_NFT[..., None] * self.covarianceDiag_NFM[:, :, None]).sum(axis=0)

            a_1 = (self.W_NFK[..., None, None] * (self.covarianceDiag_NFM[:, :, None] * (self.Qx_power_FTM / (self.Y_FTM ** 2))[None])[:, :, None] ).sum(axis=4).sum(axis=1) # N F K T M
            b_1 = (self.W_NFK[..., None, None] * (self.covarianceDiag_NFM[:, :, None] / self.Y_FTM[None])[:, :, None]).sum(axis=4).sum(axis=1) # N F K T M
            self.H_NKT = self.H_NKT * self.xp.sqrt(a_1 / b_1)
            self.lambda_NFT = self.W_NFK @ self.H_NKT
            self.Y_FTM = (self.lambda_NFT[..., None] * self.covarianceDiag_NFM[:, :, None]).sum(axis=0)


    def normalize(self):
        phi_F = self.xp.trace(self.diagonalizer_FMM @ self.diagonalizer_FMM.conj().transpose(0, 2, 1), axis1=1, axis2=2).real / self.NUM_mic
        self.diagonalizer_FMM = self.diagonalizer_FMM / self.xp.sqrt(phi_F)[:, None, None]
        self.covarianceDiag_NFM = self.covarianceDiag_NFM / phi_F[None, :, None]

        mu_NF = (self.covarianceDiag_NFM).sum(axis=2).real
        self.covarianceDiag_NFM = self.covarianceDiag_NFM / mu_NF[:, :, None]
        self.W_NFK = self.W_NFK * mu_NF[:, :, None]

        nu_NK = self.W_NFK.sum(axis=1)
        self.W_NFK = self.W_NFK / nu_NK[:, None]
        self.H_NKT = self.H_NKT * nu_NK[:, :, None]
        self.lambda_NFT = self.W_NFK @ self.H_NKT

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
    parser.add_argument(             '--gpu', type= int, default=    0, help='GPU ID')##
    parser.add_argument(           '--n_fft', type= int, default= 1024, help='number of frequencies')
    parser.add_argument(       '--NUM_noise', type= int, default=    1, help='number of noise')
    parser.add_argument(   '--NUM_iteration', type= int, default=  100, help='number of iteration')
    parser.add_argument(       '--NUM_basis', type= int, default=    8, help='number of basis')
    parser.add_argument( '--MODE_initialize_covarianceMatrix', type=  str, default="obs", help='cGMM, cGMM2, unit, obs')
    args = parser.parse_args()

    if args.gpu < 0:
        import numpy as xp
    else:
        import cupy as xp
        print("Use GPU " + str(args.gpu))
        cuda.get_device_from_id(args.gpu).use()

    wav, fs = sf.read(args.input_fileName)
    wav = wav.T
    M = len(wav)
    for m in range(M):
        tmp = librosa.core.stft(wav[m], n_fft=args.n_fft, hop_length=int(args.n_fft/4))
        if m == 0:
            spec = np.zeros([tmp.shape[0], tmp.shape[1], M], dtype=np.complex)
        spec[:, :, m] = tmp

    separater = FastMNMF(NUM_source=args.NUM_noise+1, NUM_basis=args.NUM_basis, xp=xp, MODE_initialize_covarianceMatrix=args.MODE_initialize_covarianceMatrix)
    separater.load_spectrogram(spec)
    separater.file_id = args.file_id
    separater.solve(NUM_iteration=args.NUM_iteration, save_likelihood=False, save_parameter=False, save_wav=False, save_path="./", interval_save_parameter=50)
