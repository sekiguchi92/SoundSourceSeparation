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

    def __init__(self, n_source=2, n_basis=8, xp=np, init_SCM="unit", n_bit=64, seed=0):
        """ initialize FastMNMF

        Parameters:
        -----------
            n_source: int
                The number of sources
            n_iteration: int
                The number of iteration to update all variables
            n_basis: int
                The number of bases of each source
            init_SCM: str
                How to initialize covariance matrix {circular, gradual, obs, ILRMA}
                About circular and gradual initialization, please check my paper:
                    Kouhei Sekiguchi, Yoshiaki Bando, Aditya Arie Nugraha, Kazuyoshi Yoshii, Tatsuya Kawahara:
                    Fast Multichannel Nonnegative Matrix Factorization with Directivity-Aware
                        Jointly-Diagonalizable Spatial Covariance Matrices for Blind Source Separation,
                    IEEE/ACM Transactions on Audio, Speech, and Language Processing, accepted, 2020.
            n_bit:int (32 or 64)
                The number of bits for floating point number.
                '32' may degrade the peformance in exchange for lower computational cost.
                32 -> float32 and complex64
                64 -> float64 and complex128
        """
        super(FastMNMF, self).__init__(n_source=n_source, xp=xp, init_SCM=init_SCM, n_bit=n_bit, seed=seed)
        self.method_name = "FastMNMF"
        self.n_basis = n_basis


    def initialize_PSD(self):
        self.W_NFK = self.xp.random.rand(self.n_source, self.n_freq, self.n_basis).astype(self.TYPE_FLOAT)
        self.H_NKT = self.xp.random.rand(self.n_source, self.n_basis, self.n_time).astype(self.TYPE_FLOAT)
        self.lambda_NFT = self.W_NFK @ self.H_NKT


    def make_filename_suffix(self):
        self.filename_suffix = f"S={self.n_source}-it={self.n_iteration}-L={self.n_basis}-init={self.init_SCM}"

        if self.n_bit != 64:
            self.filename_suffix += f"-bit={self.n_bit}"
        if hasattr(self, "file_id"):
            self.filename_suffix += "-ID={}".format(self.file_id)
        print("param:", self.filename_suffix)


    def update(self):
        self.update_WH()
        self.update_CovarianceDiagElement()
        self.udpate_Diagonalizer()
        self.normalize()


    def update_WH(self):
        tmp1_NFT = (self.G_NFM[:, :, None] * (self.Qx_power_FTM / (self.Y_FTM ** 2))[None]).sum(axis=3)
        tmp2_NFT = (self.G_NFM[:, :, None] / self.Y_FTM[None]).sum(axis=3)
        a_W = (self.H_NKT[:, None] * tmp1_NFT[:, :, None]).sum(axis=3)  # N F K T M
        b_W = (self.H_NKT[:, None] * tmp2_NFT[:, :, None]).sum(axis=3)
        a_H = (self.W_NFK[..., None] * tmp1_NFT[:, :, None] ).sum(axis=1) # N F K T M
        b_H = (self.W_NFK[..., None] * tmp2_NFT[:, :, None]).sum(axis=1) # N F K T M
        self.W_NFK *= self.xp.sqrt(a_W / b_W)
        self.H_NKT *= self.xp.sqrt(a_H / b_H)

        self.lambda_NFT = self.W_NFK @ self.H_NKT + EPS
        self.Y_FTM = (self.lambda_NFT[..., None] * self.G_NFM[:, :, None]).sum(axis=0)


    def normalize(self):
        phi_F = self.xp.sum(self.Q_FMM * self.Q_FMM.conj(), axis=(1, 2)).real / self.n_mic
        self.Q_FMM /= self.xp.sqrt(phi_F)[:, None, None]
        self.G_NFM /= phi_F[None, :, None]

        mu_NF = (self.G_NFM).sum(axis=2).real
        self.G_NFM /= mu_NF[:, :, None]
        self.W_NFK *= mu_NF[:, :, None]

        nu_NK = self.W_NFK.sum(axis=1)
        self.W_NFK /= nu_NK[:, None]
        self.H_NKT *= nu_NK[:, :, None]
        self.lambda_NFT = self.W_NFK @ self.H_NKT + EPS

        self.reset_variable()


    def save_parameter(self, filename):
        param_list = [self.lambda_NFT, self.G_NFM, self.Q_FMM, self.W_NFK, self.H_NKT]
        if self.xp != np:
            param_list = [self.convert_to_NumpyArray(param) for param in param_list]

        pic.dump(param_list, open(filename, "wb"))


    def load_parameter(self, filename):
        param_list = pic.load(open(filename, "rb"))
        if self.xp != np:
            param_list = [cuda.to_gpu(param) for param in param_list]

        self.lambda_NFT, self.G_NFM, self.Q_FMM, self.W_NFK, self.H_NKT = param_list



if __name__ == "__main__":
    import argparse
    import pickle as pic
    import sys, os

    parser = argparse.ArgumentParser()
    parser.add_argument('input_filename', type= str, help='filename of the multichannel observed signals')
    parser.add_argument(     '--file_id', type= str, default=    "None", help='file id')
    parser.add_argument(         '--gpu', type= int, default=         0, help='GPU ID')
    parser.add_argument(       '--n_fft', type= int, default=      1024, help='number of frequencies')
    parser.add_argument(    '--n_source', type= int, default=         4, help='number of noise sources')
    parser.add_argument(     '--n_basis', type= int, default=         8, help='number of bases')
    parser.add_argument(       '--n_mic', type= int, default=         8, help='number of microphones')
    parser.add_argument(    '--init_SCM', type= str, default= "gradual", help='circular, gradual, obs, ILRMA')
    parser.add_argument( '--n_iteration', type= int, default=       100, help='number of iteration')
    parser.add_argument(       '--n_bit', type= int, default=        64, help='number of bits for floating point number')
    args = parser.parse_args()

    if args.gpu < 0:
        import numpy as xp
    else:
        import cupy as xp
        print("Use GPU " + str(args.gpu))
        cuda.get_device_from_id(args.gpu).use()

    wav, fs = sf.read(args.input_filename)
    wav = wav.T
    M = min(args.n_mic, len(wav))
    for m in range(M):
        tmp = librosa.core.stft(wav[m], n_fft=args.n_fft, hop_length=int(args.n_fft/4))
        if m == 0:
            spec = np.zeros([tmp.shape[0], tmp.shape[1], M], dtype=np.complex)
        spec[:, :, m] = tmp

    spec += (np.random.rand(spec.shape[0], spec.shape[1], M) + np.random.rand(spec.shape[0], spec.shape[1], M) * 1j) * np.abs(spec).max() * 1e-5

    separater = FastMNMF(n_source=args.n_source, n_basis=args.n_basis, xp=xp, init_SCM=args.init_SCM, n_bit=args.n_bit, seed=0)
    separater.load_spectrogram(spec)
    separater.file_id = args.file_id
    separater.solve(n_iteration=args.n_iteration, save_likelihood=False, save_parameter=False, save_wav=False, save_path="./", interval_save_parameter=25)
