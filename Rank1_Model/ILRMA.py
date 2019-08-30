#! /usr/bin/env python3
# coding: utf-8

import numpy as np
from progressbar import progressbar
import librosa
import soundfile as sf
import matplotlib.pyplot as pl
import sys, os
import pickle as pic

sys.path.append("../CupyLibrary")
try:
    from chainer import cuda
    FLAG_GPU_Available = True
except:
    print("---Warning--- You cannot use GPU acceleration because chainer or cupy is not installed")
    FLAG_GPU_Available = False

try:
    from cupy_matrix_inverse import inv_gpu_batch
    FLAG_CupyInverse_Enabled = True
except:
    print("---Warning--- You cannot use special cupy functions")
    FLAG_CupyInverse_Enabled = False

from configure import *


class ILRMA:
    def __init__(self, NUM_basis=2, xp=np, MODE_initialize_covarianceMatrix="unit"):
        """ initialize ILRMA

        Parameters:
        -----------
            NUM_basis: int
                the number of bases of each speech source
            MODE_initialize_covarianceMatrix: str
                how to initialize covariance matrix {unit, obs, cGMM}
            xp: numpy or cupy
        """
        self.NUM_basis = NUM_basis
        self.MODE_initialize_covarianceMatrix = MODE_initialize_covarianceMatrix
        self.xp = xp
        self.calculateInverseMatrix = self.return_InverseMatrixCalculationMethod()
        self.method_name = "ILRMA"


    def load_spectrogram(self, X_FTM):
        """ load complex spectrogram

        Parameters:
        -----------
            X_FTM: self.xp.array [ F * T * M ]
                power spectrogram of observed signals
        """
        self.NUM_freq, self.NUM_time, self.NUM_mic = X_FTM.shape
        self.NUM_source = self.NUM_mic
        self.X_FTM = self.xp.asarray(X_FTM, dtype=self.xp.complex)
        self.XX_FTMM = self.X_FTM[:, :, :, None] @ self.X_FTM[:, :, None, :].conj()


    def set_parameter(self, NUM_iteration=None, MODE_initialize_covarianceMatrix=None, NUM_basis=None):
        """ set parameters

        Parameters:
        -----------
            NUM_iteration: int
                the number of iterations
            NUM_basis: int
                the number of basis
            MODE_initialize_covarianceMatrix: str
                how to initialize covariance matrix {unit, obs, cGMM, obs_all}
        """
        if NUM_iteration != None:
            self.NUM_iteration = NUM_iteration
        if NUM_basis != None:
            self.NUM_basis = NUM_basis
        if MODE_initialize_covarianceMatrix != None:
            self.MODE_initialize_covarianceMatrix = MODE_initialize_covarianceMatrix


    def convert_to_NumpyArray(self, data):
        if self.xp == np:
            return data
        else:
            return cuda.to_cpu(data)


    def return_InverseMatrixCalculationMethod(self):
        if self.xp == np:
            return np.linalg.inv
        elif FLAG_CupyInverse_Enabled:
            return inv_gpu_batch
        else:
            return lambda x: cuda.to_gpu(np.linalg.inv(self.convert_to_NumpyArray(x)))


    def initialize_parameter(self):
        power_observation_FT = (self.xp.abs(self.X_FTM).astype(self.xp.float) ** 2).mean(axis=2)
        shape = 2
        self.W_NFK = self.xp.random.gamma(shape, 1 / self.NUM_freq / shape, size=[self.NUM_source, self.NUM_freq, self.NUM_basis])
        self.W_NFK[self.W_NFK < EPS] = EPS
        self.W_NFK = self.W_NFK / self.W_NFK.sum(axis=1)[:, None]
        self.H_NKT = self.xp.random.gamma(shape, (power_observation_FT.mean() * self.NUM_freq * self.NUM_mic / (self.NUM_source * self.NUM_basis)) / shape, size=[self.NUM_source, self.NUM_basis, self.NUM_time])
        self.H_NKT[self.H_NKT < EPS] = EPS
        self.lambda_NFT = self.W_NFK @ self.H_NKT


    def initialize_covarianceMatrix(self):
        if "unit" in self.MODE_initialize_covarianceMatrix:
            mixing_matrix_FMM = self.SeparationMatrix_FMM
        elif "obs" in self.MODE_initialize_covarianceMatrix:
            power_observation = (self.xp.abs(self.X_FTM).astype(self.xp.float32) ** 2).mean(axis=2) # F T

            eig_val, eig_vector = np.linalg.eig(self.convert_to_NumpyArray(self.XX_FTMM.sum(axis=1) / power_observation.sum(axis=1)[:, None, None]  ))
            eig_vector = self.xp.asarray(eig_vector).astype(self.xp.complex)
            mixing_matrix_FMM = self.xp.zeros([self.NUM_freq, self.NUM_mic, self.NUM_mic], dtype=self.xp.complex)
            mixing_matrix_FMM[:] = self.xp.eye(self.NUM_mic).astype(self.xp.complex)
            for f in range(self.NUM_freq):
                mixing_matrix_FMM[f, :, 0] = eig_vector[f, :, eig_val[f].argmax()]

        self.SeparationMatrix_FMM = self.calculateInverseMatrix(mixing_matrix_FMM)

        mu_NF = self.xp.zeros([self.NUM_mic, self.NUM_freq])
        for m in range(self.NUM_mic):
            mu_NF[m] = (self.SeparationMatrix_FMM[:, m] * self.SeparationMatrix_FMM[:, m].conj()).sum(axis=1).real
            self.SeparationMatrix_FMM[:, m] = self.SeparationMatrix_FMM[:, m] / self.xp.sqrt(mu_NF[m, :, None])
        self.reset_variable()


    def solve(self, NUM_iteration=100, save_likelihood=False, save_parameter=False, save_wav=False, save_path="./", interval_save_parameter=25):
        """
        Parameters:
            NUM_iteration: int
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
        self.NUM_iteration = NUM_iteration
        self.initialize_covarianceMatrix()
        self.initialize_parameter()
        self.make_filename_suffix()

        log_likelihood_array = []
        for it in progressbar(range(NUM_iteration)):
            self.update()

            if save_parameter and (it > 0) and ((it+1) % interval_save_parameter == 0) and ((it+1) != NUM_iteration):
                self.save_parameter(save_path+"{}-parameter-{}-{}.pic".format(self.method_name, self.filename_suffix, it + 1))

            if save_wav and (it > 0) and ((it+1) % interval_save_parameter == 0) and ((it+1) != NUM_iteration):
                self.separate_LinearFilter(mic_index=MIC_INDEX)
                self.save_separated_signal(save_path+"{}-sep-Linear-{}-{}.wav".format(self.method_name, self.filename_suffix, it + 1))

            if save_likelihood and (it > 0) and ((it+1) % interval_save_parameter == 0) and ((it+1) != NUM_iteration):
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
        self.filename_suffix = "it={}-L={}-init={}".format(self.NUM_iteration, self.NUM_basis, self.MODE_initialize_covarianceMatrix)

        if hasattr(self, "file_id"):
           self.filename_suffix += "-ID={}".format(self.file_id)
        else:
            print("====================\n\nWarning: Please set self.file_id\n\n====================")


    def update_SeparationMatrix(self):
        """
        Results:
            self.Y_FTN: self.xp.array  [ F x T x N ]
                the separated signal in frequency domain
            self.Y_power_FTN: self.xp.array [ F x T x N ]
                the power spectrogram of the separated signal Y
        """
        for m in range(self.NUM_mic):
            V_FMM = (self.XX_FTMM / self.lambda_NFT[m, :, :, None, None]).mean(axis=1)
            tmp_FM = self.calculateInverseMatrix(self.SeparationMatrix_FMM @ V_FMM)[:, :, m]
            self.SeparationMatrix_FMM[:, m] = (tmp_FM / self.xp.sqrt(( (tmp_FM.conj()[:, :, None] * V_FMM).sum(axis=1) * tmp_FM).sum(axis=1) )[:, None]).conj()


    def update_WH(self):
        """
        Results:
            self.W_NFK: self.xp.array [ NUM_source x F x NUM_basis ]
                the template of each basis
            self.H_NKT: self.xp.array [ NUM_source x NUM_basis x T ]
                the activation of each basis
        """
        if self.xp == np:
            for f in range(self.NUM_freq):
                numerator = (self.H_NKT * (self.Y_power_FTN[f].T / ( self.lambda_NFT[:, f] ** 2 ) )[:, None]).sum(axis=2) # NUM_source * NUM_basis * F
                denominator = ( self.H_NKT / self.lambda_NFT[:, f, None] ).sum(axis=2) # NUM_source * NUM_basis * F
                self.W_NFK[:, f] = self.W_NFK[:, f] * self.xp.sqrt(numerator / denominator)
            self.W_NFK[self.W_NFK < EPS] = EPS
            self.lambda_NFT = self.W_NFK @ self.H_NKT

            numerator = self.xp.zeros_like(self.H_NKT)
            denominator = self.xp.zeros_like(self.H_NKT)
            for f in range(self.NUM_freq):
                numerator += (self.W_NFK[:, f, :, None] * (self.Y_power_FTN[f].T / ( self.lambda_NFT[:, f] ** 2 ) )[:, None]) # NUM_source * NUM_basis * T
                denominator += ( self.W_NFK[:, f, :, None] / self.lambda_NFT[:, f, None] ) # NUM_source * NUM_basis * T
            self.H_NKT = self.H_NKT * self.xp.sqrt(numerator / denominator)
            self.H_NKT[self.H_NKT < EPS] = EPS
            self.lambda_NFT = self.W_NFK @ self.H_NKT

        else:
            numerator = (self.H_NKT[:, None] * (self.Y_power_FTN.transpose(2, 0, 1) / ( self.lambda_NFT ** 2 ) )[:, :, None] ).sum(axis=3) # NUM_source * NUM_basis * F
            denominator = ( self.H_NKT[:, None] / self.lambda_NFT[:, :, None] ).sum(axis=3) # NUM_source * NUM_basis * F
            self.W_NFK = self.W_NFK * self.xp.sqrt(numerator / denominator)
            self.W_NFK[self.W_NFK < EPS] = EPS
            self.lambda_NFT = self.W_NFK @ self.H_NKT

            numerator = (self.W_NFK[:, :, :, None] * (self.Y_power_FTN.transpose(2, 0, 1) / ( self.lambda_NFT ** 2 ) )[:, :, None] ).sum(axis=1) # NUM_source * NUM_basis * T
            denominator = ( self.W_NFK[:, :, :, None] / self.lambda_NFT[:, :, None] ).sum(axis=1) # NUM_source * NUM_basis * T
            self.H_NKT = self.H_NKT * self.xp.sqrt(numerator / denominator)
            self.H_NKT[self.H_NKT < EPS] = EPS
            self.lambda_NFT = self.W_NFK @ self.H_NKT


    def normalize(self):
        mu_NF = self.xp.zeros([self.NUM_mic, self.NUM_freq])
        for m in range(self.NUM_mic):
            mu_NF[m] = (self.SeparationMatrix_FMM[:, m] * self.SeparationMatrix_FMM[:, m].conj()).sum(axis=1).real
            self.SeparationMatrix_FMM[:, m] = self.SeparationMatrix_FMM[:, m] / self.xp.sqrt(mu_NF[m, :, None])
        self.W_NFK = self.W_NFK / mu_NF[:, :, None]

        nu_NnK = self.W_NFK.sum(axis=1)
        self.W_NFK = self.W_NFK / nu_NnK[:, None]
        self.H_NKT = self.H_NKT * nu_NnK[:, :, None]

        self.lambda_NFT = self.W_NFK @ self.H_NKT
        self.reset_variable()


    def reset_variable(self):
        if self.xp == np:
            self.Y_power_FTN = self.xp.abs((self.SeparationMatrix_FMM[:, None] @ self.X_FTM[:, :, :, None] )[:, :, :, 0]) ** 2 # F * T * N
        else:
            self.Y_power_FTN = self.xp.abs((self.SeparationMatrix_FMM[:, None] * self.X_FTM[:, :, None] ).sum(axis=3) ) ** 2 # F * T * N


    def calculate_log_likelihood(self):
        return -1 * (self.Y_power_FTN.transpose(2, 0, 1) / self.lambda_NFT + self.xp.log(self.lambda_NFT)).sum() + self.NUM_time * np.log(np.linalg.det(self.convert_to_NumpyArray(self.SeparationMatrix_FMM @ self.SeparationMatrix_FMM.conj().transpose(0, 2, 1)))).sum().real


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

        self.Y_FTN = self.convert_to_NumpyArray(self.xp.matmul( self.SeparationMatrix_FMM[:, None], self.X_FTM[:, :, :, None] )[:, :, :, 0]) # F * T * N
        F, T, N = self.Y_FTN.shape
        if source_index == None:
            self.separated_spec = np.zeros([F, T, N], dtype=np.complex)
            for n in range(N):
                self.separated_spec[:, :, n] = np.matmul( self.Y_FTN[:, :, n][:, :, None], np.linalg.inv(self.convert_to_NumpyArray( self.SeparationMatrix_FMM))[:, :, n][:, None] )[:, :, mic_index]
            self.separated_spec = self.separated_spec.transpose(2, 0, 1)
        else:
            self.separated_spec = np.matmul( self.Y_FTN[:, :, source_index][:, :, None], np.linalg.inv(self.convert_to_NumpyArray( self.SeparationMatrix_FMM))[:, :, source_index][:, None] )[:, :, mic_index]
        return self.separated_spec # N F T or F T


    def save_separated_signal(self, save_fileName="sample.wav"):
        self.separated_spec = self.convert_to_NumpyArray(self.separated_spec)
        hop_length = int((self.NUM_freq - 1) / 2)
        if self.separated_spec.ndim == 2:
            separated_signal = librosa.core.istft(self.separated_spec, hop_length=hop_length)
            separated_signal /= np.abs(separated_signal).max() * 1.2
            sf.write(save_fileName, separated_signal, 16000)
        elif self.separated_spec.ndim == 3:
            for n in range(self.NUM_source):
                tmp = librosa.core.istft(self.separated_spec[n, :, :], hop_length=hop_length)
                if n == 0:
                    separated_signal = np.zeros([self.NUM_source, len(tmp)])
                separated_signal[n] = tmp
            separated_signal /= np.abs(separated_signal).max() * 1.2
            sf.write(save_fileName, separated_signal.T, 16000)


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
    parser.add_argument( 'input_fileName', type= str, help='filename of the multichannel observed signals')
    parser.add_argument(      '--file_id', type= str, default="None", help='file id')
    parser.add_argument(          '--gpu', type= int, default=     0, help='GPU ID')
    parser.add_argument(        '--n_fft', type= int, default=  1024, help='number of frequencies')
    parser.add_argument(    '--NUM_basis', type= int, default=     4, help='number of basis of noise (MODE_noise=NMF)')
    parser.add_argument('--NUM_iteration', type= int, default=    30, help='number of iteration')
    parser.add_argument( '--MODE_initialize_covarianceMatrix', type=str, default="obs2", help='cGMM, cGMM2, unit, obs')
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

    separater = ILRMA(NUM_basis=args.NUM_basis, xp=xp, MODE_initialize_covarianceMatrix=args.MODE_initialize_covarianceMatrix)
    separater.load_spectrogram(spec)
    separater.file_id = args.file_id
    separater.solve(NUM_iteration=args.NUM_iteration, save_wav=False, save_likelihood=False, save_path="./", save_parameter=False, interval_save_parameter=100)
