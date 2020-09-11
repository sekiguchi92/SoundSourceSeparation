#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import sys, os
from progressbar import progressbar
import librosa
import soundfile as sf
import pickle as pic

from configure_FastModel import *

try:
    from chainer import cuda
except:
    print("---Warning--- You cannot use GPU acceleration because chainer or cupy is not installed")


class FastFCA():
    """ Blind Source Separation Using Fast Full-rank Covariance Analysis (FastFCA)

    X_FTM: the observed complex spectrogram
    Q_FMM: diagonalizer that converts a spatial covariance matrix (SCM) to a diagonal matrix
    G_NFM: diagonal elements of the diagonalized SCMs
    lambda_NFT: power spectral densities of each source
    Qx_power_FTM: power spectra of Qx
    Y_FTM: \sum_n lambda_NFT G_NFM
    """

    def __init__(self, n_source=2, xp=np, init_SCM="unit", n_bit=64, seed=0):
        """ initialize FastFCA

        Parameters:
        -----------
            n_source: int
                The number of sources
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
        self.method_name = "FastFCA"
        self.n_source = n_source
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


    def convert_to_NumpyArray(self, data):
        if self.xp == np:
            return data
        else:
            return cuda.to_cpu(data)


    def load_spectrogram(self, X_FTM):
        """ load complex spectrogram

        Parameters:
        -----------
            X_FTM: self.xp.array [ F * T * M ]
                power spectrogram of observed signals
        """
        self.n_freq, self.n_time, self.n_mic = X_FTM.shape
        self.X_FTM = self.xp.asarray(X_FTM, dtype=self.TYPE_COMPLEX)
        self.XX_FTMM = self.X_FTM[:, :, :, None] @ self.X_FTM[:, :, None, :].conj()


    def initialize_PSD(self):
        self.lambda_NFT = self.xp.random.random([self.n_source, self.n_freq, self.n_time]).astype(self.TYPE_FLOAT)
        self.lambda_NFT[0] = self.xp.abs(self.X_FTM.mean(axis=2)) ** 2


    def initialize_covarianceMatrix(self):
        if ("unit" in self.init_SCM) or ("circular" in self.init_SCM):
            self.Q_FMM = self.xp.tile(self.xp.eye(self.n_mic), [self.n_freq, 1, 1]).astype(self.TYPE_COMPLEX)
            self.G_NFM = self.xp.ones([self.n_source, self.n_freq, self.n_mic], dtype=self.TYPE_FLOAT) * 1e-2
            for m in range(self.n_mic):
                self.G_NFM[m % self.n_source, :, m] = 1
        elif "obs" in self.init_SCM:
            mixture_covarianceMatrix_FMM = self.XX_FTMM.sum(axis=1) / (self.xp.trace(self.XX_FTMM, axis1=2, axis2=3).sum(axis=1))[:, None, None]
            eig_val, eig_vec = np.linalg.eigh(self.convert_to_NumpyArray(mixture_covarianceMatrix_FMM))
            eig_val, eig_vec = eig_val[:, ::-1], eig_vec[:, :, ::-1]
            self.Q_FMM = self.xp.asarray(eig_vec).transpose(0, 2, 1).conj()
            self.G_NFM = self.xp.ones([self.n_source, self.n_freq, self.n_mic], dtype=self.TYPE_FLOAT) * 1e-2
            self.G_NFM[0, :, 0] = 1
        elif "gradual" in self.init_SCM:
            from FastMNMF import FastMNMF
            fastmnmf1 = FastMNMF(n_source=self.n_source, n_basis=2, init_SCM="circular", xp=self.xp, n_bit=self.n_bit)
            fastmnmf1.load_spectrogram(self.X_FTM)
            fastmnmf1.solve(n_iteration=30)
            self.Q_FMM = fastmnmf1.Q_FMM
            self.G_NFM = fastmnmf1.G_NFM
        elif "ILRMA" in self.init_SCM:
            sys.path.append("../Rank1_Model")
            from ILRMA import ILRMA
            ilrma = ILRMA(n_basis=2, init_SCM="unit", xp=self.xp, n_bit=self.n_bit)
            ilrma.load_spectrogram(self.X_FTM)
            ilrma.solve(n_iteration=15, save_likelihood=False, save_wav=False, save_path="./", interval_save_parameter=1000)
            separated_spec_power = self.xp.abs(ilrma.separated_spec).mean(axis=(1, 2))
            self.Q_FMM = ilrma.SeparationMatrix_FMM
            self.G_NFM = self.xp.ones([self.n_source, self.n_freq, self.n_mic], dtype=self.TYPE_FLOAT) * 1e-2
            for n in range(self.n_source):
                self.G_NFM[n, :, separated_spec_power.argmax()] = 1
                separated_spec_power[separated_spec_power.argmax()] = 0
        else:
            raise ValueError("Please specify how to initialize covariance matrix {unit, obs}")

        self.normalize()


    def reset_variable(self):
        if self.xp == np:
            self.Qx_power_FTM = self.xp.abs((self.Q_FMM[:, None] @ self.X_FTM[:, :, :, None])[:, :, :, 0]) ** 2
        else:
            self.Qx_power_FTM = self.xp.abs((self.Q_FMM[:, None] * self.X_FTM[:, :, None]).sum(axis=3)) ** 2
        self.Y_FTM = (self.lambda_NFT[..., None] * self.G_NFM[:, :, None]).sum(axis=0)


    def make_filename_suffix(self):
        self.filename_suffix = f"S={self.n_source}-it={self.n_iteration}-init={self.init_SCM}"

        if self.n_bit != 64:
            self.filename_suffix += f"-bit={self.n_bit}"
        if hasattr(self, "file_id"):
            self.filename_suffix += f"-ID={self.file_id}"
        print("param:", self.filename_suffix)


    def solve(self, n_iteration=100, save_likelihood=False, save_parameter=False, save_wav=False, save_path="./", interval_save_parameter=30, mic_index=MIC_INDEX):
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
        self.n_iteration = n_iteration

        self.initialize_PSD()
        self.initialize_covarianceMatrix()
        self.make_filename_suffix()

        log_likelihood_array = []
        for it in progressbar(range(self.n_iteration)):
            self.update()

            if save_parameter and ((it+1) % interval_save_parameter == 0) and ((it+1) != self.n_iteration):
                self.save_parameter(save_path+"{}-parameter-{}-{}.pic".format(self.method_name, self.filename_suffix, it + 1))

            if save_wav and ((it+1) % interval_save_parameter == 0) and ((it+1) != self.n_iteration):
                self.separate_FastWienerFilter(mic_index=MIC_INDEX)
                self.save_separated_signal(save_path+"{}-sep-Wiener-{}-{}.wav".format(self.method_name, self.filename_suffix, it + 1))

            if save_likelihood and ((it+1) % interval_save_parameter == 0) and ((it+1) != self.n_iteration):
                log_likelihood_array.append(self.calculate_log_likelihood())

        if save_parameter:
            self.save_parameter(save_path+"{}-parameter-{}.pic".format(self.method_name, self.filename_suffix))

        if save_likelihood:
            log_likelihood_array.append(self.calculate_log_likelihood())
            pic.dump(log_likelihood_array, open(save_path + "{}-likelihood-interval={}-{}.pic".format(self.method_name, interval_save_parameter, self.filename_suffix), "wb"))

        self.separate_FastWienerFilter(mic_index=mic_index)
        self.save_separated_signal(save_path+"{}-sep-Wiener-{}.wav".format(self.method_name, self.filename_suffix))


    def update(self):
        self.update_lambda()
        self.update_CovarianceDiagElement()
        self.udpate_Diagonalizer()
        self.normalize()


    def udpate_Diagonalizer(self):
        for m in range(self.n_mic):
            V_FMM = (self.XX_FTMM / self.Y_FTM[:, :, m, None, None]).mean(axis=1)
            tmp_FM = self.xp.linalg.inv(self.Q_FMM @ V_FMM)[:, :, m]
            self.Q_FMM[:, m] = (tmp_FM / self.xp.sqrt(( (tmp_FM.conj()[:, :, None] * V_FMM).sum(axis=1) * tmp_FM).sum(axis=1) )[:, None]).conj()


    def update_CovarianceDiagElement(self):
        a_1 = (self.lambda_NFT[..., None] * (self.Qx_power_FTM / (self.Y_FTM ** 2))[None]).sum(axis=2) # N F T M
        b_1 = (self.lambda_NFT[..., None] / self.Y_FTM[None]).sum(axis=2)
        self.G_NFM *= self.xp.sqrt(a_1 / b_1)
        self.G_NFM += EPS
        self.Y_FTM = (self.lambda_NFT[..., None] * self.G_NFM[:, :, None]).sum(axis=0)


    def update_lambda(self):
        a = (self.G_NFM[:, :, None] * (self.Qx_power_FTM / (self.Y_FTM ** 2))[None]).sum(axis=3) # N F T
        b = (self.G_NFM[:, :, None] / self.Y_FTM[None]).sum(axis=3)
        self.lambda_NFT *= self.xp.sqrt(a / b) 
        self.lambda_NFT += EPS
        self.Y_FTM = (self.lambda_NFT[..., None] * self.G_NFM[:, :, None]).sum(axis=0)


    def normalize(self):
        phi_F = self.xp.sum(self.Q_FMM * self.Q_FMM.conj(), axis=(1, 2)).real / self.n_mic
        self.Q_FMM = self.Q_FMM / self.xp.sqrt(phi_F)[:, None, None]
        self.G_NFM = self.G_NFM / phi_F[None, :, None]

        mu_NF = (self.G_NFM).sum(axis=2).real
        self.G_NFM = self.G_NFM / mu_NF[:, :, None]
        self.lambda_NFT = self.lambda_NFT * mu_NF[:, :, None] + EPS

        self.reset_variable()


    def calculate_log_likelihood(self):
        return (-(self.Qx_power_FTM / self.Y_FTM).sum() + self.n_time * np.log(np.linalg.det(self.convert_to_NumpyArray(self.Q_FMM @ self.Q_FMM.conj().transpose(0, 2, 1) ) ) ).sum() - self.xp.log(self.Y_FTM).sum()).real - self.n_mic * self.n_freq * self.n_time * np.log(np.pi)


    def calculate_covarianceMatrix(self):
        covarianceMatrix_NFMM = self.xp.zeros([self.n_source, self.n_freq, self.n_mic, self.n_mic], dtype=self.TYPE_COMPLEX)
        diagonalizer_inv_FMM = self.xp.linalg.inv(self.Q_FMM)
        for n in range(self.n_source):
            for f in range(self.n_freq):
                covarianceMatrix_NFMM[n, f] = diagonalizer_inv_FMM[f] @ np.diag(self.G_NFM[n, f]) @ diagonalizer_inv_FMM[f].conj().T
        return covarianceMatrix_NFMM


    def separate_FastWienerFilter(self, source_index=None, mic_index=MIC_INDEX):
        Qx_FTM = (self.Q_FMM[:, None] * self.X_FTM[:, :, None]).sum(axis=3)
        diagonalizer_inv_FMM = self.xp.linalg.inv(self.Q_FMM)
        if source_index != None:
            self.separated_spec = self.convert_to_NumpyArray((diagonalizer_inv_FMM[:, None] @ (Qx_FTM * ( (self.lambda_NFT[source_index, :, :, None] * self.G_NFM[source_index, :, None]) / (self.lambda_NFT[..., None]* self.G_NFM[:, :, None]).sum(axis=0) ) )[..., None])[:, :, mic_index, 0])
        else:
            for n in range(self.n_source):
                tmp = self.convert_to_NumpyArray((diagonalizer_inv_FMM[:, None] @ (Qx_FTM * ( (self.lambda_NFT[n, :, :, None] * self.G_NFM[n, :, None]) / (self.lambda_NFT[..., None]* self.G_NFM[:, :, None]).sum(axis=0) ) )[..., None])[:, :, mic_index, 0])
                if n == 0:
                    self.separated_spec = np.zeros([self.n_source, tmp.shape[0], tmp.shape[1]], dtype=np.complex)
                self.separated_spec[n] = tmp


    def save_separated_signal(self, save_filename="sample.wav"):
        self.separated_spec = self.convert_to_NumpyArray(self.separated_spec)
        hop_length = int((self.n_freq - 1) / 2)
        if self.separated_spec.ndim == 2:
            separated_signal = librosa.core.istft(self.separated_spec, hop_length=hop_length)
            separated_signal /= np.abs(separated_signal).max() * 1.2
            sf.write(save_filename, separated_signal, 16000)
        elif self.separated_spec.ndim == 3:
            for n in range(self.n_source):
                tmp = librosa.core.istft(self.separated_spec[n, :, :], hop_length=hop_length)
                if n == 0:
                    separated_signal = np.zeros([self.n_source, len(tmp)])
                separated_signal[n] = tmp
            separated_signal /= np.abs(separated_signal).max() * 1.2
            sf.write(save_filename, separated_signal.T, 16000)


    def save_parameter(self, filename):
        param_list = [self.lambda_NFT, self.G_NFM, self.Q_FMM]
        if self.xp != np:
            param_list = [self.convert_to_NumpyArray(param) for param in param_list]
        pic.dump(param_list, open(filename, "wb"))


    def load_parameter(self, filename):
        param_list = pic.load(open(filename, "rb"))
        if self.xp != np:
            param_list = [cuda.to_gpu(param) for param in param_list]
        self.lambda_NFT, self.G_NFM, self.Q_FMM = param_list

        self.n_source, self.n_freq, self.n_time = self.lambda_NFT.shape
        self.n_mic = self.G_NFM.shape[-1]


if __name__ == "__main__":
    import sys, os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input_filename', type= str, help='filename of the multichannel observed signals')
    parser.add_argument(     '--file_id', type= str, default=    "None", help='file id')
    parser.add_argument(         '--gpu', type= int, default=         0, help='GPU ID')
    parser.add_argument(       '--n_fft', type= int, default=      1024, help='number of frequencies')
    parser.add_argument(    '--n_source', type= int, default=         2, help='number of noise')
    parser.add_argument(    '--init_SCM', type= str, default= "gradual", help='circular, gradual, obs, ILRMA')
    parser.add_argument( '--n_iteration', type= int, default=       100, help='number of iteration')
    parser.add_argument(       '--n_mic', type= int, default=         8, help='number of microphones')
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

    separater = FastFCA(n_source=args.n_source, xp=xp, init_SCM=args.init_SCM, n_bit=args.n_bit, seed=0)
    separater.load_spectrogram(spec)
    separater.file_id = args.file_id
    separater.solve(n_iteration=args.n_iteration, save_likelihood=False, save_parameter=False, save_wav=False, save_path="./", interval_save_parameter=25)
