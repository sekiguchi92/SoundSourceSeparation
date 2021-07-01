#! /usr/bin/env python3
# coding: utf-8

import numpy as np
from progressbar import progressbar
import librosa
import soundfile as sf
import matplotlib.pyplot as pl
import sys, os
import pickle as pic

from configure import *


class ILRMA:
    """ Blind Source Separation Using Independent Low-rank Matrix Analysis (ILRMA)

    X_FTM: the observed complex spectrogram
    W_NFK: basis vectors for each source
    H_NKT: activations for each source
    lambda_NFT: power spectral densities of each source (W_NFK @ H_NKT)
    SeparationMatrix_FMM: separation matrices
    """

    def __init__(self, n_basis=2, xp=np, init_SCM="unit", n_bit=64, seed=0):
        """ initialize ILRMA

        Parameters:
        -----------
            n_basis: int
                the number of bases of each speech source
            init_SCM: str
                how to initialize covariance matrix {unit, obs}
            xp: numpy or cupy
        """
        self.method_name = "ILRMA"
        self.n_basis = n_basis
        self.init_SCM = init_SCM
        self.xp = xp
        self.xp.random.seed(seed)
        np.random.seed(seed)

        self.n_bit = n_bit
        if self.n_bit == 32:
            self.TYPE_FLOAT = self.xp.float32
            self.TYPE_COMPLEX = self.xp.complex64
        elif self.n_bit == 64:
            self.TYPE_FLOAT = self.xp.float64
            self.TYPE_COMPLEX = self.xp.complex128
        else:
            raise ValueError("n_bit must be 32 or 64")


    def load_spectrogram(self, X_FTM):
        """ load complex spectrogram

        Parameters:
        -----------
            X_FTM: self.xp.array [ F * T * M ]
                power spectrogram of observed signals
        """
        self.n_freq, self.n_time, self.n_mic = X_FTM.shape
        self.n_source = self.n_mic
        self.X_FTM = self.xp.asarray(X_FTM, dtype=self.TYPE_COMPLEX)
        self.XX_FTMM = self.X_FTM[:, :, :, None] @ self.X_FTM[:, :, None, :].conj()


    def convert_to_NumpyArray(self, data):
        if self.xp == np:
            return data
        else:
            return cuda.to_cpu(data)


    def initialize_PSD(self):
        self.W_NFK = self.xp.random.rand(self.n_source, self.n_freq, self.n_basis).astype(self.TYPE_FLOAT)
        self.H_NKT = self.xp.random.rand(self.n_source, self.n_basis, self.n_time).astype(self.TYPE_FLOAT)
        self.lambda_NFT = self.W_NFK @ self.H_NKT


    def initialize_covarianceMatrix(self):
        if "unit" in self.init_SCM:
            mixing_matrix_FMM = self.xp.tile(self.xp.eye(self.n_mic), [self.n_freq, 1, 1]).astype(self.TYPE_COMPLEX)
        elif "obs" in self.init_SCM:
            power_observation = (self.xp.abs(self.X_FTM).astype(self.xp.float32) ** 2).mean(axis=2) # F T

            eig_val, eig_vector = np.linalg.eig(self.convert_to_NumpyArray(self.XX_FTMM.sum(axis=1) / power_observation.sum(axis=1)[:, None, None]  ))
            eig_vector = self.xp.asarray(eig_vector).astype(self.TYPE_COMPLEX)
            mixing_matrix_FMM = self.xp.zeros([self.n_freq, self.n_mic, self.n_mic], dtype=self.TYPE_COMPLEX)
            mixing_matrix_FMM[:] = self.xp.eye(self.n_mic).astype(self.TYPE_COMPLEX)
            for f in range(self.n_freq):
                mixing_matrix_FMM[f, :, 0] = eig_vector[f, :, eig_val[f].argmax()]

        self.SeparationMatrix_FMM = self.xp.linalg.inv(mixing_matrix_FMM)
        self.normalize()


    def solve(self, n_iteration=100, save_likelihood=False, save_parameter=False, save_wav=False, save_path="./", interval_save_parameter=25):
        """
        Parameters:
            n_iteration: int
                the number of iteration to update all variables
            save_likelihood: boolean
                save likelihood and lower bound or not
            save_parameter: boolean
                save parameter or not
            save_path: str
                directory for saving data
            interval_save_parameter: int
                interval of saving parameter
        """
        self.n_iteration = n_iteration
        self.initialize_PSD()
        self.initialize_covarianceMatrix()
        self.make_filename_suffix()

        log_likelihood_array = []
        for it in progressbar(range(n_iteration)):
            self.update()

            if save_parameter and (it > 0) and ((it+1) % interval_save_parameter == 0) and ((it+1) != n_iteration):
                self.save_parameter(save_path+"{}-parameter-{}-{}.pic".format(self.method_name, self.filename_suffix, it + 1))

            if save_wav and (it > 0) and ((it+1) % interval_save_parameter == 0) and ((it+1) != n_iteration):
                self.separate_LinearFilter(mic_index=MIC_INDEX)
                self.save_separated_signal(save_path+"{}-sep-Linear-{}-{}.wav".format(self.method_name, self.filename_suffix, it + 1))

            if save_likelihood and (it > 0) and ((it+1) % interval_save_parameter == 0) and ((it+1) != n_iteration):
                log_likelihood_array.append(self.calculate_log_likelihood())

        if save_likelihood:
            log_likelihood_array.append(self.calculate_log_likelihood())
            pic.dump(log_likelihood_array, open(save_path + "{}-likelihood-interval={}-{}.pic".format(self.method_name, interval_save_parameter, self.filename_suffix), "wb"))

        if save_parameter:
            self.save_parameter(save_path+"{}-parameter-{}.pic".format(self.method_name, self.filename_suffix))

        self.separate_LinearFilter(mic_index=MIC_INDEX)
        self.save_separated_signal(save_path+"{}-sep-Linear-{}.wav".format(self.method_name, self.filename_suffix))


    def update(self):
        self.update_WH()
        self.update_SeparationMatrix()
        self.normalize()


    def make_filename_suffix(self):
        self.filename_suffix = "it={}-L={}-init={}".format(self.n_iteration, self.n_basis, self.init_SCM)

        if self.n_bit != 64:
            self.filename_suffix += f"-bit={self.n_bit}"
        if hasattr(self, "file_id"):
           self.filename_suffix += "-ID={}".format(self.file_id)
        print("param:", self.filename_suffix)


    def update_SeparationMatrix(self):
        """
        Results:
            self.Y_FTN: self.xp.array  [ F x T x N ]
                the separated signal in frequency domain
            self.Y_power_FTN: self.xp.array [ F x T x N ]
                the power spectrogram of the separated signal Y
        """
        for m in range(self.n_mic):
            V_FMM = (self.XX_FTMM / self.lambda_NFT[m, :, :, None, None]).mean(axis=1)
            tmp_FM = self.xp.linalg.inv(self.SeparationMatrix_FMM @ V_FMM)[:, :, m]
            self.SeparationMatrix_FMM[:, m] = (tmp_FM / self.xp.sqrt(( (tmp_FM.conj()[:, :, None] * V_FMM).sum(axis=1) * tmp_FM).sum(axis=1) )[:, None]).conj()


    def update_WH(self):
        """
        Results:
            self.W_NFK: self.xp.array [ n_source x F x n_basis ]
                the template of each basis
            self.H_NKT: self.xp.array [ n_source x n_basis x T ]
                the activation of each basis
        """
        if self.xp == np:
            for f in range(self.n_freq):
                numerator = (self.H_NKT * (self.Y_power_FTN[f].T / ( self.lambda_NFT[:, f] ** 2 ) )[:, None]).sum(axis=2) # n_source * n_basis * F
                denominator = ( self.H_NKT / self.lambda_NFT[:, f, None] ).sum(axis=2) # n_source * n_basis * F
                self.W_NFK[:, f] = self.W_NFK[:, f] * self.xp.sqrt(numerator / denominator)
            self.W_NFK[self.W_NFK < EPS] = EPS
            self.lambda_NFT = self.W_NFK @ self.H_NKT

            numerator = self.xp.zeros_like(self.H_NKT)
            denominator = self.xp.zeros_like(self.H_NKT)
            for f in range(self.n_freq):
                numerator += (self.W_NFK[:, f, :, None] * (self.Y_power_FTN[f].T / ( self.lambda_NFT[:, f] ** 2 ) )[:, None]) # n_source * n_basis * T
                denominator += ( self.W_NFK[:, f, :, None] / self.lambda_NFT[:, f, None] ) # n_source * n_basis * T
            self.H_NKT = self.H_NKT * self.xp.sqrt(numerator / denominator)
            self.H_NKT[self.H_NKT < EPS] = EPS
            self.lambda_NFT = self.W_NFK @ self.H_NKT

        else:
            numerator = (self.H_NKT[:, None] * (self.Y_power_FTN.transpose(2, 0, 1) / ( self.lambda_NFT ** 2 ) )[:, :, None] ).sum(axis=3) # n_source * n_basis * F
            denominator = ( self.H_NKT[:, None] / self.lambda_NFT[:, :, None] ).sum(axis=3) # n_source * n_basis * F
            self.W_NFK = self.W_NFK * self.xp.sqrt(numerator / denominator)
            self.W_NFK[self.W_NFK < EPS] = EPS
            self.lambda_NFT = self.W_NFK @ self.H_NKT

            numerator = (self.W_NFK[:, :, :, None] * (self.Y_power_FTN.transpose(2, 0, 1) / ( self.lambda_NFT ** 2 ) )[:, :, None] ).sum(axis=1) # n_source * n_basis * T
            denominator = ( self.W_NFK[:, :, :, None] / self.lambda_NFT[:, :, None] ).sum(axis=1) # n_source * n_basis * T
            self.H_NKT = self.H_NKT * self.xp.sqrt(numerator / denominator)
            self.H_NKT[self.H_NKT < EPS] = EPS
            self.lambda_NFT = self.W_NFK @ self.H_NKT


    def normalize(self):
        mu_NF = self.xp.zeros([self.n_mic, self.n_freq], dtype=self.TYPE_FLOAT)
        for m in range(self.n_mic):
            mu_NF[m] = (self.SeparationMatrix_FMM[:, m] * self.SeparationMatrix_FMM[:, m].conj()).sum(axis=1).real
            self.SeparationMatrix_FMM[:, m] /= self.xp.sqrt(mu_NF[m, :, None])
        self.W_NFK /= mu_NF[:, :, None]

        nu_NnK = self.W_NFK.sum(axis=1)
        self.W_NFK /= nu_NnK[:, None]
        self.H_NKT *= nu_NnK[:, :, None]

        self.lambda_NFT = self.W_NFK @ self.H_NKT
        self.reset_variable()


    def reset_variable(self):
        if self.xp == np:
            self.Y_power_FTN = self.xp.abs((self.SeparationMatrix_FMM[:, None] @ self.X_FTM[:, :, :, None] )[:, :, :, 0]) ** 2 # F * T * N
        else:
            self.Y_power_FTN = self.xp.abs((self.SeparationMatrix_FMM[:, None] * self.X_FTM[:, :, None] ).sum(axis=3) ) ** 2 # F * T * N


    def calculate_log_likelihood(self):
        return -1 * (self.Y_power_FTN.transpose(2, 0, 1) / self.lambda_NFT + self.xp.log(self.lambda_NFT)).sum() + self.n_time * np.log(np.linalg.det(self.convert_to_NumpyArray(self.SeparationMatrix_FMM @ self.SeparationMatrix_FMM.conj().transpose(0, 2, 1)))).sum().real


    def separate_LinearFilter(self, source_index=None, mic_index=MIC_INDEX):
        """ return separated spectrograms

        Parameters:
        ----------
            source_index: int
                the index of the source. If None, the separated spectrograms of all sources are returned (default is None)
            mic_index: int
                the index of the microphone of which you want to get the source image 
        Returns:
            separated_spec: numpy.ndarray
                If source_index is None, shape is N x F x T, and else F x T
        """
        self.Y_FTN = ( self.SeparationMatrix_FMM[:, None] @ self.X_FTM[:, :, :, None] ).squeeze()
        if source_index == None:
            self.separated_spec = self.xp.zeros([self.n_mic, self.n_freq, self.n_time], dtype=self.TYPE_COMPLEX)
            for n in range(self.n_mic):
                self.separated_spec[n] = self.Y_FTN[:, :, n] * self.xp.linalg.inv(self.SeparationMatrix_FMM)[:, None, mic_index, n]
        else:
            self.separated_spec = self.Y_FTN[:, :, source_index] @ self.xp.linalg.inv(self.SeparationMatrix_FMM)[:, None, mic_index, source_index]
        return self.separated_spec # N F T or F T


    def save_separated_signal(self, save_filename="sample.wav"):
        separated_spec = self.convert_to_NumpyArray(self.separated_spec)
        hop_length = int((self.n_freq - 1) / 2)
        if separated_spec.ndim == 2:
            separated_signal = librosa.core.istft(separated_spec, hop_length=hop_length)
            separated_signal /= np.abs(separated_signal).max() * 1.2
            sf.write(save_filename, separated_signal, 16000)
        elif separated_spec.ndim == 3:
            for n in range(self.n_source):
                tmp = librosa.core.istft(separated_spec[n, :, :], hop_length=hop_length)
                if n == 0:
                    separated_signal = np.zeros([self.n_source, len(tmp)])
                separated_signal[n] = tmp
            separated_signal /= np.abs(separated_signal).max() * 1.2
            sf.write(save_filename, separated_signal.T, 16000)


    def save_parameter(self, filename):
        param_list = [self.SeparationMatrix_FMM, self.lambda_NFT]
        param_list.append(self.W_NFK)
        param_list.append(self.H_NKT)

        if self.xp != np:
            param_list = [self.convert_to_NumpyArray(param) for param in param_list]

        pic.dump(param_list, open(filename, "wb"))


    def load_parameter(self, filename):
        param_list = pic.load(open(filename, "rb"))

        if self.xp != np:
            param_list = [self.xp.asarray(param) for param in param_list]

        self.SeparationMatrix_FMM = param_list[0]
        self.lambda_NFT = param_list[1]
        self.W_NFK = param_list[2]
        self.H_NKT = param_list[3]



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument( 'input_filename', type= str, help='filename of the multichannel observed signals')
    parser.add_argument(      '--file_id', type= str, default="None", help='file id')
    parser.add_argument(          '--gpu', type= int, default=     0, help='GPU ID')
    parser.add_argument(        '--n_fft', type= int, default=  1024, help='number of frequencies')
    parser.add_argument(      '--n_basis', type= int, default=     4, help='number of basis of noise (MODE_noise=NMF)')
    parser.add_argument(  '--n_iteration', type= int, default=    30, help='number of iteration')
    parser.add_argument(     '--init_SCM', type= str, default= "obs", help='unit, obs')
    parser.add_argument(        '--n_bit', type= int, default=    64, help='number of bits for floating point number')
    args = parser.parse_args()

    if args.gpu < 0:
        import numpy as xp
    else:
        import cupy as xp
        print("Use GPU " + str(args.gpu))
        xp.cuda.Device(args.gpu).use()

    wav, fs = sf.read(args.input_filename)
    wav = wav.T
    M = len(wav)
    for m in range(M):
        tmp = librosa.core.stft(wav[m], n_fft=args.n_fft, hop_length=int(args.n_fft/4))
        if m == 0:
            spec = np.zeros([tmp.shape[0], tmp.shape[1], M], dtype=np.complex)
        spec[:, :, m] = tmp

    separater = ILRMA(n_basis=args.n_basis, xp=xp, init_SCM=args.init_SCM, n_bit=args.n_bit, seed=0)
    separater.load_spectrogram(spec)
    separater.file_id = args.file_id
    separater.solve(n_iteration=args.n_iteration, save_wav=False, save_likelihood=False, save_path="./", save_parameter=False, interval_save_parameter=100)
