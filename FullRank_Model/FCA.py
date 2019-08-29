#! /usr/bin/env python3
# coding: utf-8

import sys, os
import numpy as np
from progressbar import progressbar
import librosa
import soundfile as sf
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
    print("---Warning--- You cannot use cupy inverse calculation")
    FLAG_CupyInverse_Enabled = False

try:
    from cupy_eig import det_Hermitian
    FLAG_CupyDeterminant_Enabled = True
except:
    print("---Warning--- You cannot use cupy complex determinant")
    FLAG_CupyDeterminant_Enabled = False

from geometric_mean import geometric_mean_invA
from configure import *


class FCA:

    def __init__(self, NUM_source=2, xp=np, MODE_initialize_covarianceMatrix="unit", MODE_update_parameter=["all", "one_by_one"][0]):
        """ initialize FCA

        Parameters:
        -----------
            NUM_source: int
                the number of sources
            NUM_iteration: int
                the number of iteration to update all variables
            xp : numpy or cupy
            MODE_initialize_covarianceMatrix: str
                how to initialize covariance matrix {unit, obs, cGMM}
            MODE_update_parameter: str
                'all' : update all the parameters simultanesouly to reduce computational cost
                'one_by_one' : update the parameters one by one to monotonically increase log-likelihood
        """
        self.NUM_source = NUM_source
        self.MODE_initialize_covarianceMatrix = MODE_initialize_covarianceMatrix
        self.MODE_update_parameter = MODE_update_parameter
        self.xp = xp
        self.calculateInverseMatrix = self.return_InverseMatrixCalculationMethod()
        self.method_name = "FCA"


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
            return lambda x: cuda.to_gpu(np.linalg.inv(convert_to_NumpyArray(x)))


    def set_parameter(self, NUM_iteration=None, NUM_source=None, MODE_initialize_covarianceMatrix=None, MODE_update_parameter=None):
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
        if NUM_iteration != None:
            self.NUM_iteration = NUM_iteration
        if NUM_source != None:
            self.NUM_source = NUM_source
        if MODE_initialize_covarianceMatrix != None:
            self.MODE_initialize_covarianceMatrix = MODE_initialize_covarianceMatrix
        if MODE_update_parameter != None:
            self.MODE_update_parameter = MODE_update_parameter


    def load_spectrogram(self, X_FTM):
        """ load complex spectrogram

        Parameters:
        -----------
            X_FTM: self.xp.array [ F * T * M ]
                power spectrogram of observed signals
        """
        self.NUM_freq, self.NUM_time, self.NUM_mic = X_FTM.shape
        self.X_FTM = self.xp.asarray(X_FTM, dtype=self.xp.complex)
        self.XX_FTMM = self.xp.matmul( self.X_FTM[:, :, :, None], self.xp.conj( self.X_FTM[:, :, None, :] ) ) # F T M M


    def initialize_covarianceMatrix(self):
        self.covarianceMatrix_NFMM = self.xp.zeros([self.NUM_source, self.NUM_freq, self.NUM_mic, self.NUM_mic], dtype=self.xp.complex)
        self.covarianceMatrix_NFMM[:, :] = self.xp.eye(self.NUM_mic).astype(self.xp.complex)
        if "unit" in self.MODE_initialize_covarianceMatrix:
            pass
        elif "obs" in self.MODE_initialize_covarianceMatrix:
            power_observation_FT = (self.xp.abs(self.X_FTM).astype(self.xp.float) ** 2).mean(axis=2) # F T
            self.covarianceMatrix_NFMM[0] = self.XX_FTMM.sum(axis=1) / power_observation_FT.sum(axis=1)[:, None, None] # F M M

        self.covarianceMatrix_NFMM = self.covarianceMatrix_NFMM / self.xp.trace(self.covarianceMatrix_NFMM, axis1=2 ,axis2=3)[:, :, None, None]


    def initialize_PSD(self):
        self.lambda_NFT = self.xp.random.random([self.NUM_source, self.NUM_freq, self.NUM_time]).astype(self.xp.float)
        self.lambda_NFT[0] = self.xp.abs(self.X_FTM.mean(axis=2)) ** 2


    def solve(self, NUM_iteration=100, save_likelihood=False, save_parameter=False, save_wav=False, save_dir="./", interval_save_parameter=30):
        """
        Parameters:
            save_likelihood: boolean
                save likelihood and lower bound or not
            save_parameter: boolean
                save parameter or not
            save_wav: boolean
                save intermediate separated signal or not
            save_dir: str
                directory for saving data
            interval_save_parameter: int
                interval of saving parameter
        """
        self.NUM_iteration = NUM_iteration

        self.initialize_covarianceMatrix()
        self.initialize_PSD()
        self.make_filename_suffix()

        log_likelihood_array = []
        for it in progressbar(range(self.NUM_iteration)):
            self.update()

            if save_parameter and (it > 0) and ((it+1) % interval_save_parameter == 0) and ((it+1) != self.NUM_iteration):
                self.save_parameter(save_dir+"{}-parameter-{}-{}.pic".format(self.method_name, self.filename_suffix, it + 1))

            if save_wav and (it > 0) and ((it+1) % interval_save_parameter == 0) and ((it+1) != self.NUM_iteration):
                self.separate_WienerFilter(mic_index=MIC_INDEX)
                self.save_separated_signal(save_dir+"{}-sep-Wiener-{}-{}.wav".format(self.method_name, self.filename_suffix, it + 1))

            if save_likelihood and (it > 0) and ((it+1) % interval_save_parameter == 0) and ((it+1) != self.NUM_iteration):
                log_likelihood_array.append(self.calculate_log_likelihood())

        if save_parameter:
            self.save_parameter(save_dir+"{}-parameter-{}.pic".format(self.method_name, self.filename_suffix))

        if save_likelihood:
            log_likelihood_array.append(self.calculate_log_likelihood())
            pic.dump(log_likelihood_array, open(save_dir + "{}-likelihood-interval={}-{}.pic".format(self.method_name, interval_save_parameter, self.filename_suffix), "wb"))

        self.separate_WienerFilter(mic_index=MIC_INDEX)
        self.save_separated_signal(save_dir+"{}-sep-Wiener-{}.wav".format(self.method_name, self.filename_suffix))


    def make_filename_suffix(self):
        self.filename_suffix = "S={}-it={}-init={}-update={}".format(self.NUM_source, self.NUM_iteration, self.MODE_initialize_covarianceMatrix, self.MODE_update_parameter)

        if hasattr(self, "file_id"):
           self.filename_suffix += "-ID={}".format(self.file_id)
        else:
            print("====================\n\nWarning: Please set self.file_id\n\n====================")
        return self.filename_suffix


    def update(self):
        if self.MODE_update_parameter == "one_by_one":
            self.update_axiliary_variable()
            self.update_lambda()
            self.update_axiliary_variable()
            self.update_covarianceMatrix()
        if self.MODE_update_parameter == "all":
            self.update_axiliary_variable()
            self.update_lambda()
            self.update_covarianceMatrix()
        self.normalize()


    def update_axiliary_variable(self):
        self.Yinv_FTMM = self.calculateInverseMatrix( (self.lambda_NFT[..., None, None] * self.covarianceMatrix_NFMM[:, :, None]).sum(axis=0) )

        Yx_FTM1 = self.Yinv_FTMM @ self.X_FTM[..., None]
        self.Yinv_X_Yinv_FTMM = Yx_FTM1 @ Yx_FTM1.conj().transpose(0, 1, 3, 2)
        self.tr_Cov_Yinv_X_Yinv_NFT = self.xp.trace(self.covarianceMatrix_NFMM[:, :, None] @ self.Yinv_X_Yinv_FTMM[None], axis1=3, axis2=4).real
        self.tr_Cov_Yinv_NFT = self.xp.trace(self.covarianceMatrix_NFMM[:, :, None] @ self.Yinv_FTMM[None], axis1=3, axis2=4).real


    def update_lambda(self):
        self.lambda_NFT = self.lambda_NFT * self.xp.sqrt(self.tr_Cov_Yinv_X_Yinv_NFT / self.tr_Cov_Yinv_NFT)


    def update_covarianceMatrix(self):
        a_1 = (self.lambda_NFT[..., None, None] * self.Yinv_FTMM[None]).sum(axis=2)# + (self.xp.eye(self.NUM_mic) * EPS)[None, None]  # N F M M
        b_1 = self.covarianceMatrix_NFMM @ (self.lambda_NFT[..., None, None] * self.Yinv_X_Yinv_FTMM[None]).sum(axis=2) @ self.covarianceMatrix_NFMM + (self.xp.eye(self.NUM_mic) * EPS)[None, None] # N F M M
        self.covarianceMatrix_NFMM = geometric_mean_invA(a_1, b_1, xp=self.xp)
        self.covarianceMatrix_NFMM = (self.covarianceMatrix_NFMM + self.covarianceMatrix_NFMM.transpose(0, 1, 3, 2).conj()) / 2 # for stability

    def normalize(self):
        mu_NF = self.xp.trace(self.covarianceMatrix_NFMM, axis1=2, axis2=3).real
        self.covarianceMatrix_NFMM = self.covarianceMatrix_NFMM / mu_NF[:, :, None, None]
        self.lambda_NFT = self.lambda_NFT * mu_NF[:, :, None]


    def calculate_log_likelihood(self):
        if self.xp == np:
            Yinv_FTMM = np.linalg.inv((self.lambda_NFT[..., None, None] * self.covarianceMatrix_NFMM[:, :, None]).sum(axis=0))
            return (- np.trace(Yinv_FTMM @ self.XX_FTMM, axis1=2, axis2=3).real + np.log(np.linalg.det(Yinv_FTMM).real)).sum()
        elif FLAG_CupyDeterminant_Enabled:
            Yinv_FTMM = self.calculateInverseMatrix((self.lambda_NFT[..., None, None] * self.covarianceMatrix_NFMM[:, :, None]).sum(axis=0))
            return (- self.xp.trace(Yinv_FTMM @ self.XX_FTMM, axis1=2, axis2=3).real + self.xp.log(det_Hermitian(Yinv_FTMM))).sum()
        else:
            Yinv_FTMM = self.calculateInverseMatrix((self.lambda_NFT[..., None, None] * self.covarianceMatrix_NFMM[:, :, None]).sum(axis=0))
            return (- self.xp.trace(Yinv_FTMM @ self.XX_FTMM, axis1=2, axis2=3).real).sum() + np.log(np.linalg.det(self.convert_to_NumpyArray(Yinv_FTMM))).sum()


    def separate_WienerFilter(self, mic_index=MIC_INDEX): # separate using Wiener filter
        Omega_NFTMM = (self.lambda_NFT[:, :, :, None, None] * self.covarianceMatrix_NFMM[:, :, None]) # N F T M M
        Omega_sum_inv_FTMM = self.calculateInverseMatrix(Omega_NFTMM.sum(axis=0)) # F T M M
        self.separated_spec = self.convert_to_NumpyArray(((Omega_NFTMM @ Omega_sum_inv_FTMM[None]) @ self.X_FTM[None, :, :, :, None])[:, :, :, mic_index, 0])
        return self.separated_spec


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
        param_list = [self.covarianceMatrix_NFMM, self.lambda_NFT]

        if self.xp != np:
            param_list = [cuda.to_cpu(param) for param in param_list]

        pic.dump(param_list, open(filename, "wb"))


    def load_parameter(self, filename):
        param_list = pic.load(open(filename, "rb"))
        if self.xp != np:
            param_list = [cuda.to_gpu(param) for param in param_list]
        self.covarianceMatrix_NFMM = param_list[0]
        self.lambda_NFT = param_list[1]

        self.NUM_source, self.NUM_freq, self.NUM_time = self.lambda_NFT.shape
        self.NUM_mic = self.covarianceMatrix_NFMM.shape[3]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(    'input_fileName', type= str, help='filename of the multichannel observed signals')
    parser.add_argument(                         '--file_id', type= str, default="None", help='file id')
    parser.add_argument(                             '--gpu', type= int, default=     0, help='GPU ID')
    parser.add_argument(                           '--n_fft', type= int, default=  1024, help='number of frequencies')
    parser.add_argument(                      '--NUM_source', type= int, default=     2, help='number of noise')
    parser.add_argument(                   '--NUM_iteration', type= int, default=    30, help='number of iteration')
    parser.add_argument(            '--MODE_update_parameter', type= str, default= "all", help='all, one_by_one')
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

    separater = FCA(NUM_source = args.NUM_source, xp=xp, MODE_initialize_covarianceMatrix=args.MODE_initialize_covarianceMatrix, MODE_update_parameter=args.MODE_update_parameter)
    separater.load_spectrogram(spec)
    separater.file_id = args.file_id
    separater.solve(NUM_iteration=args.NUM_iteration, save_likelihood=False, save_parameter=False, save_dir="./", interval_save_parameter=300)
