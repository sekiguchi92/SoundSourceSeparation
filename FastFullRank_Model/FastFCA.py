#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import sys, os
from progressbar import progressbar
import librosa
import soundfile as sf
import pickle as pic

from configure_FastModel import *

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


class FastFCA():

    def __init__(self, NUM_source=2, xp=np, MODE_initialize_covarianceMatrix="unit"):
        """ initialize FastFCA

        Parameters:
        -----------
            NUM_source: int
                the number of sources
            MODE_initialize_covarianceMatrix: str
                how to initialize covariance matrix {unit, obs, cGMM}
        """
        self.NUM_source = NUM_source
        self.MODE_initialize_covarianceMatrix = MODE_initialize_covarianceMatrix
        self.xp = xp
        self.calculateInverseMatrix = self.return_InverseMatrixCalculationMethod()
        self.method_name = "FastFCA"


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


    def set_parameter(self, NUM_source=None, MODE_initialize_covarianceMatrix=None):
        """ set parameters

        Parameters:
        -----------
            NUM_source: int
                the number of sources
            MODE_initialize_covarianceMatrix: str
                how to initialize covariance matrix {unit, obs, cGMM}
        """
        if NUM_source != None:
            self.NUM_source = NUM_source
        if MODE_initialize_covarianceMatrix != None:
            self.MODE_initialize_covarianceMatrix = MODE_initialize_covarianceMatrix


    def load_spectrogram(self, X_FTM):
        """ load complex spectrogram

        Parameters:
        -----------
            X_FTM: self.xp.array [ F * T * M ]
                power spectrogram of observed signals
        """
        self.NUM_freq, self.NUM_time, self.NUM_mic = X_FTM.shape
        self.X_FTM = self.xp.asarray(X_FTM, dtype=self.xp.complex)
        self.XX_FTMM = self.X_FTM[:, :, :, None] @ self.X_FTM[:, :, None, :].conj()
        self.lambda_NFT = self.xp.random.random([self.NUM_source, self.NUM_freq, self.NUM_time]).astype(self.xp.float)
        self.covarianceDiag_NFM = self.xp.ones([self.NUM_source, self.NUM_freq, self.NUM_mic], dtype=self.xp.float) / self.NUM_mic
        self.diagonalizer_FMM = self.xp.zeros([self.NUM_freq, self.NUM_mic, self.NUM_mic], dtype=self.xp.complex)
        self.diagonalizer_FMM[:] = self.xp.eye(self.NUM_mic).astype(self.xp.complex)


    def check_parameter(self):
        param_list = [self.lambda_NFT, self.covarianceDiag_NFM, self.diagonalizer_FMM]
        param_name_list = ["lambda_NFT", "covarianceDiag_NFM", "diagonalizer_FMM"]
        flag = 0
        for i, param in enumerate(param_list):
            if self.xp.isnan(param).any():
                flag = 1
                print("Error : " + param_name_list[i] + " have nan")
            elif self.xp.isinf(param).any():
                flag = 1
                print("Error : " + param_name_list[i] + " have inf")
        return flag


    def initialize_PSD(self):
        self.lambda_NFT[0] = self.xp.abs(self.X_FTM.mean(axis=2)) ** 2
        self.reset_variable()


    def initialize_covarianceMatrix(self):
        covarianceMatrix_NFMM = self.xp.zeros([self.NUM_source, self.NUM_freq, self.NUM_mic, self.NUM_mic], dtype=self.xp.complex)
        covarianceMatrix_NFMM[:, :] = self.xp.eye(self.NUM_mic).astype(self.xp.complex)
        if "unit" in self.MODE_initialize_covarianceMatrix:
            pass
        elif "obs" in self.MODE_initialize_covarianceMatrix:
            power_observation_FT = (self.xp.abs(self.X_FTM).astype(self.xp.float) ** 2).mean(axis=2) # F T
            covarianceMatrix_NFMM[0] = self.XX_FTMM.sum(axis=1) / power_observation_FT.sum(axis=1)[:, None, None] # F M M
        else:
            print("Please specify how to initialize covariance matrix {unit, obs}")
            raise ValueError

        covarianceMatrix_NFMM = covarianceMatrix_NFMM / self.xp.trace(covarianceMatrix_NFMM, axis1=2 ,axis2=3)[:, :, None, None]
        H_FMM = self.convert_to_NumpyArray(self.calculateInverseMatrix(covarianceMatrix_NFMM[1] @ covarianceMatrix_NFMM[0]))
        eig_val, eig_vec = np.linalg.eig(H_FMM)
        self.diagonalizer_FMM = self.xp.asarray(eig_vec.transpose(0, 2, 1).conj())
        for f in range(self.NUM_freq):
            for n in range(self.NUM_source):
                self.covarianceDiag_NFM[n, f] = self.xp.asarray(self.xp.diag(self.diagonalizer_FMM[f] @ covarianceMatrix_NFMM[n, f] @ self.diagonalizer_FMM[f].T.conj()).real)
        self.normalize()
        self.reset_variable()


    def reset_variable(self):
        if self.xp == np:
            self.Qx_power_FTM = self.xp.abs((self.diagonalizer_FMM[:, None] @ self.X_FTM[:, :, :, None])[:, :, :, 0]) ** 2
        else:
            self.Qx_power_FTM = self.xp.abs((self.diagonalizer_FMM[:, None] * self.X_FTM[:, :, None]).sum(axis=3)) ** 2
        self.Y_FTM = (self.lambda_NFT[..., None] * self.covarianceDiag_NFM[:, :, None]).sum(axis=0)


    def make_fileName_suffix(self):
        self.fileName_suffix = "S={}-it={}-init={}".format(self.NUM_source, self.NUM_iteration, self.MODE_initialize_covarianceMatrix)

        if hasattr(self, "file_id"):
            self.fileName_suffix += "-ID={}".format(self.file_id)
        else:
            print("====================\n\nWarning: Please set self.file_id\n\n====================")

        print("parameter:", self.fileName_suffix)
        return self.fileName_suffix


    def solve(self, NUM_iteration=100, save_likelihood=False, save_parameter=False, save_wav=False, save_path="./", interval_save_parameter=30, mic_index=MIC_INDEX):
        """
        Parameters:
            save_likelihood: boolean
                save likelihood and lower bound or not
            save_parameter: boolean
                save parameter or not
            save_wav: boolean
                save intermediate separated signal or not
            save_path: str
                directory for saving data
            interval_save_parameter: int
                interval of saving parameter
        """
        self.NUM_iteration = NUM_iteration

        self.initialize_PSD()
        self.initialize_covarianceMatrix()
        self.make_fileName_suffix()

        log_likelihood_array = []
        for it in progressbar(range(self.NUM_iteration)):
            self.update()

            if save_parameter and ((it+1) % interval_save_parameter == 0) and ((it+1) != self.NUM_iteration):
                self.save_parameter(save_path+"{}-parameter-{}-{}.pic".format(self.method_name, self.fileName_suffix, it + 1))

            if save_wav and ((it+1) % interval_save_parameter == 0) and ((it+1) != self.NUM_iteration):
                self.separate_FastWienerFilter(mic_index=MIC_INDEX)
                self.save_separated_signal(save_path+"{}-sep-Wiener-{}-{}.wav".format(self.method_name, self.fileName_suffix, it + 1))

            if save_likelihood and ((it+1) % interval_save_parameter == 0) and ((it+1) != self.NUM_iteration):
                log_likelihood_array.append(self.calculate_log_likelihood())

        flag = self.check_parameter()
        if flag == 1:
            print("Error --- some parameters include Nan ---")
            raise ValueError

        if save_parameter:
            self.save_parameter(save_path+"{}-parameter-{}.pic".format(self.method_name, self.fileName_suffix))

        if save_likelihood:
            log_likelihood_array.append(self.calculate_log_likelihood())
            pic.dump(log_likelihood_array, open(save_path + "{}-likelihood-interval={}-{}.pic".format(self.method_name, interval_save_parameter, self.fileName_suffix), "wb"))

        self.separate_FastWienerFilter(mic_index=mic_index)
        self.save_separated_signal(save_path+"{}-sep-Wiener-{}.wav".format(self.method_name, self.fileName_suffix))


    def update(self):
        self.update_lambda()
        self.update_CovarianceDiagElement()
        self.udpate_Diagonalizer()
        self.normalize()


    def udpate_Diagonalizer(self):
        V_FMMM = (self.XX_FTMM[:, :, None] / self.Y_FTM[:, :, :, None, None]).mean(axis=1)
        for m in range(self.NUM_mic):
            tmp_FM = self.calculateInverseMatrix(self.diagonalizer_FMM @ V_FMMM[:, m])[:, :, m]
            self.diagonalizer_FMM[:, m] = (tmp_FM / self.xp.sqrt(( (tmp_FM.conj()[:, :, None] * V_FMMM[:, m]).sum(axis=1) * tmp_FM).sum(axis=1) )[:, None]).conj()


    def update_CovarianceDiagElement(self):
        a_1 = (self.lambda_NFT[..., None] * (self.Qx_power_FTM / (self.Y_FTM ** 2))[None]).sum(axis=2) # N F T M
        b_1 = (self.lambda_NFT[..., None] / self.Y_FTM[None]).sum(axis=2)
        self.covarianceDiag_NFM = self.covarianceDiag_NFM * self.xp.sqrt(a_1 / b_1)
        self.covarianceDiag_NFM += EPS
        self.Y_FTM = (self.lambda_NFT[..., None] * self.covarianceDiag_NFM[:, :, None]).sum(axis=0)


    def update_lambda(self):
        a = (self.covarianceDiag_NFM[:, :, None] * (self.Qx_power_FTM / (self.Y_FTM ** 2))[None]).sum(axis=3) # N F T
        b = (self.covarianceDiag_NFM[:, :, None] / self.Y_FTM[None]).sum(axis=3)
        self.lambda_NFT = self.lambda_NFT * self.xp.sqrt(a / b)
        self.Y_FTM = (self.lambda_NFT[..., None] * self.covarianceDiag_NFM[:, :, None]).sum(axis=0)


    def normalize(self):
        phi_F = self.xp.sum(self.diagonalizer_FMM * self.diagonalizer_FMM.conj(), axis=(1, 2)).real / self.NUM_mic
        self.diagonalizer_FMM = self.diagonalizer_FMM / self.xp.sqrt(phi_F)[:, None, None]
        self.covarianceDiag_NFM = self.covarianceDiag_NFM / phi_F[None, :, None]

        mu_NF = (self.covarianceDiag_NFM).sum(axis=2).real
        self.covarianceDiag_NFM = self.covarianceDiag_NFM / mu_NF[:, :, None]
        self.lambda_NFT = self.lambda_NFT * mu_NF[:, :, None]

        self.reset_variable()


    def calculate_log_likelihood(self):
        return (-(self.Qx_power_FTM / self.Y_FTM).sum() + self.NUM_time * np.log(np.linalg.det(self.convert_to_NumpyArray(self.diagonalizer_FMM @ self.diagonalizer_FMM.conj().transpose(0, 2, 1) ) ) ).sum() - self.xp.log(self.Y_FTM).sum()).real - self.NUM_mic * self.NUM_freq * self.NUM_time * np.log(np.pi)


    def calculate_covarianceMatrix(self):
        covarianceMatrix_NFMM = self.xp.zeros([self.NUM_source, self.NUM_freq, self.NUM_mic, self.NUM_mic], dtype=self.xp.complex)
        diagonalizer_inv_FMM = self.calculateInverseMatrix(self.diagonalizer_FMM)
        for n in range(self.NUM_source):
            for f in range(self.NUM_freq):
                covarianceMatrix_NFMM[n, f] = diagonalizer_inv_FMM[f] @ np.diag(self.covarianceDiag_NFM[n, f]) @ diagonalizer_inv_FMM[f].conj().T
        return covarianceMatrix_NFMM


    def separate_FastWienerFilter(self, source_index=None, mic_index=MIC_INDEX):
        Qx_FTM = (self.diagonalizer_FMM[:, None] * self.X_FTM[:, :, None]).sum(axis=3)
        if source_index != None:
            diagonalizer_inv_FMM = self.calculateInverseMatrix(self.diagonalizer_FMM)
            self.separated_spec = self.convert_to_NumpyArray((diagonalizer_inv_FMM[:, None] @ (Qx_FTM * ( (self.lambda_NFT[source_index, :, :, None] * self.covarianceDiag_NFM[source_index, :, None]) / (self.lambda_NFT[..., None]* self.covarianceDiag_NFM[:, :, None]).sum(axis=0) ) )[..., None])[:, :, mic_index, 0])
        else:
            for n in range(self.NUM_source):
                diagonalizer_inv_FMM = self.calculateInverseMatrix(self.diagonalizer_FMM)
                tmp = self.convert_to_NumpyArray((diagonalizer_inv_FMM[:, None] @ (Qx_FTM * ( (self.lambda_NFT[n, :, :, None] * self.covarianceDiag_NFM[n, :, None]) / (self.lambda_NFT[..., None]* self.covarianceDiag_NFM[:, :, None]).sum(axis=0) ) )[..., None])[:, :, mic_index, 0])
                if n == 0:
                    self.separated_spec = np.zeros([self.NUM_source, tmp.shape[0], tmp.shape[1]], dtype=np.complex)
                self.separated_spec[n] = tmp


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


    def save_parameter(self, fileName):
        param_list = [self.lambda_NFT, self.covarianceDiag_NFM, self.diagonalizer_FMM]
        if self.xp != np:
            param_list = [self.convert_to_NumpyArray(param) for param in param_list]
        pic.dump(param_list, open(fileName, "wb"))


    def load_parameter(self, fileName):
        param_list = pic.load(open(fileName, "rb"))
        if self.xp != np:
            param_list = [cuda.to_gpu(param) for param in param_list]
        self.lambda_NFT, self.covarianceDiag_NFM, self.diagonalizer_FMM = param_list

        self.NUM_source, self.NUM_freq, self.NUM_time = self.lambda_NFT.shape
        self.NUM_mic = self.covarianceDiag_NFM.shape[-1]


if __name__ == "__main__":
    import sys, os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(    'input_fileName', type= str, help='filename of the multichannel observed signals')
    parser.add_argument(         '--file_id', type= str, default="None", help='file id')
    parser.add_argument(             '--gpu', type=  int, default=    0, help='GPU ID')
    parser.add_argument(           '--n_fft', type=  int, default= 1024, help='number of frequencies')
    parser.add_argument(      '--NUM_source', type=  int, default=    2, help='number of noise')
    parser.add_argument(   '--NUM_iteration', type=  int, default=  100, help='number of iteration')
    parser.add_argument(       '--NUM_basis', type=  int, default=    8, help='number of basis')
    parser.add_argument( '--MODE_initialize_covarianceMatrix', type=  str, default="obs", help='unit, obs')
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

    separater = FastFCA(NUM_source=args.NUM_source, xp=xp, MODE_initialize_covarianceMatrix=args.MODE_initialize_covarianceMatrix)
    separater.load_spectrogram(spec)
    separater.file_id = file_id
    separater.solve(NUM_iteration=args.NUM_iteration, save_likelihood=False, save_parameter=False, save_wav=False, save_path="./", interval_save_parameter=25)
