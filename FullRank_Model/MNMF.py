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

    def __init__(self, NUM_source=2, NUM_basis=2, xp=np, MODE_initialize_covarianceMatrix="unit", MODE_update_parameter=["all", "one_by_one"][0]):
        """ initialize MNMF

        Parameters:
        -----------
            NUM_source: int
                the number of sources
            NUM_basis: int
                the number of bases of each source
            xp : numpy or cupy
            MODE_initialize_covarianceMatrix: str
                how to initialize covariance matrix {unit, obs, cGMM}
            MODE_update_parameter: str
                'all' : update all the parameters simultanesouly to reduce computational cost
                'one_by_one' : update the parameters one by one to monotonically increase log-likelihood
        """
        super(MNMF, self).__init__(NUM_source=NUM_source, xp=xp, MODE_initialize_covarianceMatrix=MODE_initialize_covarianceMatrix, MODE_update_parameter=MODE_update_parameter)
        self.NUM_basis = NUM_basis
        self.method_name = "MNMF"


    def set_parameter(self, NUM_source=None, NUM_iteration=None, NUM_basis=None, MODE_initialize_covarianceMatrix=None, MODE_update_parameter=None):
        """ set parameters

        Parameters:
        -----------
            NUM_source: int
                the number of sources
            MODE_initialize_covarianceMatrix: str
                how to initialize covariance matrix {unit, obs, cGMM}
            MODE_update_parameter: str
                'all' : update all the variables simultanesouly
                'one_by_one' : update one by one
        """
        if NUM_source != None:
            self.NUM_source = NUM_source
        if NUM_iteration != None:
            self.NUM_iteration = NUM_iteration
        if NUM_basis != None:
            self.NUM_basis = NUM_basis
        if MODE_initialize_covarianceMatrix != None:
            self.MODE_initialize_covarianceMatrix = MODE_initialize_covarianceMatrix
        if MODE_update_parameter != None:
            self.MODE_update_parameter = MODE_update_parameter
    

    def initialize_PSD(self):
        power_observation_FT = (self.xp.abs(self.X_FTM).astype(self.xp.float) ** 2).mean(axis=2)
        shape = 2
        self.W_NFK = self.xp.random.dirichlet(np.ones(self.NUM_freq)*shape, size=[self.NUM_source, self.NUM_basis]).transpose(0, 2, 1)
        self.H_NKT = self.xp.random.gamma(shape, (power_observation_FT.mean() * self.NUM_freq * self.NUM_mic / (self.NUM_source * self.NUM_basis)) / shape, size=[self.NUM_source, self.NUM_basis, self.NUM_time])
        self.H_NKT[self.H_NKT < EPS] = EPS

        self.lambda_NFT = self.W_NFK @ self.H_NKT


    def make_filename_suffix(self):
        self.filename_suffix = "S={}-it={}-K={}-init={}-update={}".format(self.NUM_source, self.NUM_iteration, self.NUM_basis, self.MODE_initialize_covarianceMatrix, self.MODE_update_parameter)

        if hasattr(self, "file_id"):
           self.filename_suffix += "-ID={}".format(self.file_id)
        else:
            print("====================\n\nWarning: Please set self.file_id\n\n====================")


    def update(self):
        if self.MODE_update_parameter == "one_by_one":
            self.update_axiliary_variable()
            self.update_W()
            self.update_axiliary_variable()
            self.update_H()
            self.update_axiliary_variable()
            self.update_covarianceMatrix()
        if self.MODE_update_parameter == "all":
            self.update_axiliary_variable()
            self.update_WH()
            self.update_covarianceMatrix()
        self.normalize()


    def update_WH(self):
        a_1 = (self.H_NKT.transpose(0, 2, 1)[:, None] * self.tr_Cov_Yinv_X_Yinv_NFT[:, :, :, None]).sum(axis=2) # Nn F K
        b_1 = (self.H_NKT.transpose(0, 2, 1)[:, None] * self.tr_Cov_Yinv_NFT[:, :, :, None]).sum(axis=2) # Nn F K

        a_2 = (self.W_NFK[..., None] * self.tr_Cov_Yinv_X_Yinv_NFT[:, :, None]).sum(axis=1) # Nn K T
        b_2 = (self.W_NFK[..., None] * self.tr_Cov_Yinv_NFT[:, :, None]).sum(axis=1) # Nn K T

        self.W_NFK = self.W_NFK * self.xp.sqrt(a_1 / b_1)
        self.H_NKT = self.H_NKT * self.xp.sqrt(a_2 / b_2)


    def update_W(self):
        a_1 = (self.H_NKT.transpose(0, 2, 1)[:, None] * self.tr_Cov_Yinv_X_Yinv_NFT[:, :, :, None]).sum(axis=2) # Nn F K
        b_1 = (self.H_NKT.transpose(0, 2, 1)[:, None] * self.tr_Cov_Yinv_NFT[:, :, :, None]).sum(axis=2) # Nn F K
        self.W_NFK = self.W_NFK * self.xp.sqrt(a_1 / b_1)
        self.lambda_NFT = self.W_NFK @ self.H_NKT


    def update_H(self):
        a_1 = (self.W_NFK[..., None] * self.tr_Cov_Yinv_X_Yinv_NFT[:, :, None]).sum(axis=1) # Nn K T
        b_1 = (self.W_NFK[..., None] * self.tr_Cov_Yinv_NFT[:, :, None]).sum(axis=1) # Nn K T
        self.H_NKT = self.H_NKT * self.xp.sqrt(a_1 / b_1)
        self.lambda_NFT = self.W_NFK @ self.H_NKT


    def normalize(self):
        mu_NF = self.xp.trace(self.covarianceMatrix_NFMM, axis1=2, axis2=3).real
        self.covarianceMatrix_NFMM = self.covarianceMatrix_NFMM / mu_NF[:, :, None, None]
        self.W_NFK = self.W_NFK * mu_NF[:, :, None]

        nu_NK = self.W_NFK.sum(axis=1)
        self.W_NFK = self.W_NFK / nu_NK[:, None]
        self.H_NKT = self.H_NKT * nu_NK[:, :, None]

        self.lambda_NFT = self.W_NFK @ self.H_NKT


    def save_parameter(self, filename):
        param_list = [self.covarianceMatrix_NFMM, self.lambda_NFT]
        param_list.append(self.W_NFK)
        param_list.append(self.H_NKT)

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
    parser.add_argument(        'input_fileName', type= str, help='filename of the multichannel observed signals')
    parser.add_argument(             '--file_id', type= str, default="None", help='file id')
    parser.add_argument(                 '--gpu', type= int, default=     0, help='GPU ID')
    parser.add_argument(               '--n_fft', type= int, default=  1024, help='number of frequencies')

    parser.add_argument(           '--NUM_basis', type= int, default=    16, help='number of basis of NMF')
    parser.add_argument(          '--NUM_source', type= int, default=     2, help='number of noise')
    parser.add_argument(       '--NUM_iteration', type= int, default=    30, help='number of iteration')
    parser.add_argument('--MODE_update_parameter', type= str, default= "all", help='all, one_by_one')
    parser.add_argument('--MODE_initialize_covarianceMatrix', type= str, default= "obs", help='cGMM, unit, obs')
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

    separater = MNMF(NUM_source=args.NUM_source, NUM_basis=args.NUM_basis, xp=xp, MODE_initialize_covarianceMatrix=args.MODE_initialize_covarianceMatrix, MODE_update_parameter=args.MODE_update_parameter)
    separater.load_spectrogram(spec)
    separater.file_id = args.file_id
    separater.solve(NUM_iteration=args.NUM_iteration, save_likelihood=False, save_parameter=False, save_dir="./", interval_save_parameter=100)
