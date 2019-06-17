#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import sys, os
from progressbar import progressbar
import librosa
import soundfile as sf
import time
import pickle as pic

sys.path.append("/home/sekiguch/Insync/program/python/my_python_library")
sys.path.append("../my_python_library")
sys.path.append("/home/sekiguch/Insync/program/python/fast_fullrank_model")
import separation
from configure import *
from geometric_mean import geometric_mean_invA
from calculate_separation_performance import calculate_separation_performance, calculate_SDR_SIR_SAR

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

try:
    from cupy_eig import det_Hermitian
    FLAG_CupyDeterminant_Enabled = True
except:
    print("---Warning--- You cannot use cupy complex determinant")
    FLAG_CupyDeterminant_Enabled = False


class FastFCA():

    def __init__(self, NUM_source=2, xp=np, MODE_initialize_covarianceMatrix="unit"):
        """ initialize FastFCA

        Parameters:
        -----------
            NUM_source: int
                the number of sources
            NUM_iteration: int
                the number of iteration to update all variables
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
            #     # tmp_FM = np.linalg.inv(self.diagonalizer_FMM @ V_FMM)[:, :, m]
            #     tmp_FM = np.linalg.solve(self.diagonalizer_FMM @ V_FMM, self.xp.eye(self.NUM_mic)[None, m])
        elif FLAG_CupyInverse_Enabled:
            return inv_gpu_batch
        else:
            return lambda x: chainer.cuda.to_gpu(np.linalg.inv(convert_to_NumpyArray(x)))


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
        elif "cGMM" in self.MODE_initialize_covarianceMatrix:
            flag = 0
            try:
                pic_filename = DIR_CGMM + "cGMM-parameter-N=2-it=100-init=obs-ID={}-25.pic".format(self.file_id)
            except:
                print("please set self.file_id")
                flag = 1

            if (flag == 0) and os.path.isfile(pic_filename):
                print("Use cGMM pic file : " + pic_filename + " for initialize covariance matrix")
                covarianceMatrix_cGMM_NFMM, power_NFT, mixingCoefficient_NT, responsibility_NFT = pic.load(open(pic_filename, "rb"))
                if self.xp != np:
                    responsibility_NFT = cuda.to_gpu(responsibility_NFT)
            else:
                sys.path.append("/home/sekiguch/Dropbox/program/python/cGMM/")
                sys.path.append("../cGMM/")
                from cGMM import cGMM
                cgmm = cGMM(self.X_FTM, NUM_source=2, xp=self.xp)
                cgmm.file_id = self.file_id
                cgmm.solve(25, save_parameter=False)
                responsibility_NFT = cgmm.responsibility_NFT
            covarianceMatrix_NFMM[0] = (responsibility_NFT[0][:, :, None, None] * self.XX_FTMM).sum(axis=1)
            covarianceMatrix_NFMM[1:] = (responsibility_NFT[1][:, :, None, None] * self.XX_FTMM).sum(axis=1)[None]
        elif "ILRMA" in self.MODE_initialize_covarianceMatrix:
            print("spatial covariance matrix is initialized by using ILRMA")
            sys.path.append("../rank1_model")
            from ILRMA import ILRMA
            ilrma = ILRMA(X=self.X_FTM, NUM_basis=2, xp=self.xp, mode_initialize_covarianceMatrix="unit")
            ilrma.file_id = self.file_id
            ilrma.solve(NUM_iteration=30, save_likelihood=False, save_parameter=False, save_wav=True, save_dir="./", interval_save_parameter=1000)
            if self.NUM_mic == self.NUM_source:
                StearingVector_FMM = self.calculateInverseMatrix(ilrma.SeparationMatrix_FMM)
                for m in range(self.NUM_mic):
                    covarianceMatrix_NFMM[m] = StearingVector_FMM[:, :, m, None] @ StearingVector_FMM[:, :, m][:, None].conj() + EPS * self.xp.eye(self.NUM_mic)[None]
            else:
                raise ValueError
        else:
            print("Please specify how to initialize covariance matrix {unit, obs, cGMM}")
            raise ValueError

        if "ILRMA" in self.MODE_initialize_covarianceMatrix:
            self.diagonalizer_FMM = ilrma.SeparationMatrix_FMM
        else:
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


    def make_filename_suffix(self):
        self.filename_suffix = "S={}-it={}-init={}".format(self.NUM_source, self.NUM_iteration, self.MODE_initialize_covarianceMatrix)

        if hasattr(self, "file_id"):
            self.filename_suffix += "-ID={}".format(self.file_id)
        else:
            print("====================\n\nWarning: Please set self.file_id\n\n====================")

        print("filename_suffix:", self.filename_suffix)
        return self.filename_suffix


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

        self.initialize_PSD()
        self.initialize_covarianceMatrix()
        self.make_filename_suffix()

        elapsedTime = 0
        log_likelihood_array = []
        for it in progressbar(range(self.NUM_iteration)):
            self.it = it
            start = time.time()

            self.update()

            elapsedTime += time.time() - start

            if save_parameter and ((it+1) % interval_save_parameter == 0) and ((it+1) != self.NUM_iteration):
                self.save_parameter(save_dir+"{}-parameter-{}-{}.pic".format(self.method_name, self.filename_suffix, it + 1))

            if save_wav and ((it+1) % interval_save_parameter == 0) and ((it+1) != self.NUM_iteration):
                self.separate_FastWienerFilter(mic_index=MIC_INDEX)
                self.save_separated_signal(save_dir+"{}-sep-Wiener-{}-{}.wav".format(self.method_name, self.filename_suffix, it + 1))

            if save_likelihood and ((it+1) % interval_save_parameter == 0) and ((it+1) != self.NUM_iteration):
                log_likelihood_array.append(self.calculate_log_likelihood())

        flag = self.check_parameter()
        if flag == 1:
            raise ValueError

        if save_parameter:
            self.save_parameter(save_dir+"{}-parameter-{}.pic".format(self.method_name, self.filename_suffix))

        if save_likelihood:
            log_likelihood_array.append(self.calculate_log_likelihood())
            pic.dump(log_likelihood_array, open(save_dir + "{}-likelihood-interval={}-{}.pic".format(self.method_name, interval_save_parameter, self.filename_suffix), "wb"))

        # self.separate_FastWienerFilter(mic_index=MIC_INDEX)
        self.separate_FastWienerFilter(mic_index=0)
        self.save_separated_signal(save_dir+"{}-sep-Wiener-{}.wav".format(self.method_name, self.filename_suffix))

        return elapsedTime


    def update(self):
        self.update_lambda()
        self.update_CovarianceDiagElement()
        self.udpate_Diagonalizer()
        self.normalize()


    def udpate_Diagonalizer(self):
        # V_FMMM = (self.XX_FTMM[:, :, None] / self.Y_FTM[:, :, :, None, None]).mean(axis=1)
        V_FMMM = (self.XX_FTMM[:, :, None] / self.Y_FTM[..., None, None] ).mean(axis=1)
        for m in range(self.NUM_mic):
            tmp_FM = self.calculateInverseMatrix(self.diagonalizer_FMM @ V_FMMM[:, m])[:, :, m]
            self.diagonalizer_FMM[:, m] = (tmp_FM / self.xp.sqrt(( (tmp_FM.conj()[:, :, None] * V_FMMM[:, m]).sum(axis=1) * tmp_FM).sum(axis=1) )[:, None]).conj()


    # def udpate_Diagonalizer(self):
    #     for m in range(self.NUM_mic):
    #         V_FMM = (self.XX_FTMM / self.Y_FTM[:, :, m, None, None]).mean(axis=1)
    #         tmp_FM = self.calculateInverseMatrix(self.diagonalizer_FMM @ V_FMM)[:, :, m]
    #         self.diagonalizer_FMM[:, m] = (tmp_FM / self.xp.sqrt(( (tmp_FM.conj()[:, :, None] * V_FMM).sum(axis=1) * tmp_FM).sum(axis=1) )[:, None]).conj()


    def udpate_Diagonalizer_FPI(self):
        for it in range(self.NUM_Q_iteration):
            for m in range(self.NUM_mic):
                self.diagonalizer_FMM[:, m] = (self.calculateInverseMatrix((self.XX_FTMM / (self.Y_FTM)[:, :, m, None, None]).mean(axis=1)) * self.calculateInverseMatrix(self.diagonalizer_FMM)[:, :, m][:, None]).sum(axis=2).conj()


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
        phi_F = self.xp.trace(self.diagonalizer_FMM @ self.diagonalizer_FMM.conj().transpose(0, 2, 1), axis1=1, axis2=2).real / self.NUM_mic
        self.diagonalizer_FMM = self.diagonalizer_FMM / self.xp.sqrt(phi_F)[:, None, None]
        self.covarianceDiag_NFM = self.covarianceDiag_NFM / phi_F[None, :, None]

        mu_NF = (self.covarianceDiag_NFM).sum(axis=2).real
        self.covarianceDiag_NFM = self.covarianceDiag_NFM / mu_NF[:, :, None]
        self.lambda_NFT = self.lambda_NFT * mu_NF[:, :, None]

        self.reset_variable()


    def calculate_log_likelihood(self):
        if FLAG_CupyDeterminant_Enabled:
            return (-(self.Qx_power_FTM / self.Y_FTM).sum() + self.NUM_time * self.xp.log(det_Hermitian(self.diagonalizer_FMM @ self.diagonalizer_FMM.conj().transpose(0, 2, 1) ) ).sum() - self.xp.log(self.Y_FTM).sum()).real - self.NUM_mic * self.NUM_freq * self.NUM_time * self.xp.log(self.xp.pi)
        else:
            return (-(self.Qx_power_FTM / self.Y_FTM).sum() + self.NUM_time * np.log(np.linalg.det(self.convert_to_NumpyArray(self.diagonalizer_FMM @ self.diagonalizer_FMM.conj().transpose(0, 2, 1) ) ) ).sum() - self.xp.log(self.Y_FTM).sum()).real - self.NUM_mic * self.NUM_freq * self.NUM_time * np.log(np.pi)


    def calculate_covarianceMatrix(self):
        covarianceMatrix_NFMM = self.xp.zeros([self.NUM_source, self.NUM_freq, self.NUM_mic, self.NUM_mic], dtype=self.xp.complex)
        diagonalizer_inv_FMM = self.calculateInverseMatrix(self.diagonalizer_FMM)
        for n in range(self.NUM_source):
            for f in range(self.NUM_freq):
                covarianceMatrix_NFMM[n, f] = diagonalizer_inv_FMM[f] @ np.diag(self.covarianceDiag_NFM[n, f]) @ diagonalizer_inv_FMM[f].conj().T
        return covarianceMatrix_NFMM


    def separate_WienerFilter(self, mic_index=MIC_INDEX): # separate using winner filter
        covarianceMatrix_NFMM = self.calculate_covarianceMatrix()
        self.separated_spec = separation.separate_MultiWienerFilter(X_FTM=self.X_FTM, power_NFT=self.lambda_NFT, covarianceMatrix_NFMM=covarianceMatrix_NFMM, inverse_function=self.calculateInverseMatrix, mic_index=mic_index)


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


    def separate_by_Q(self, source_index=None, mic_index=MIC_INDEX):
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

        self.Y_FTN = self.convert_to_NumpyArray(self.xp.matmul( self.diagonalizer_FMM[:, None], self.X_FTM[:, :, :, None] )[:, :, :, 0]) # F * T * N
        F, T, N = self.Y_FTN.shape
        if source_index == None:
            self.separated_spec = np.zeros([F, T, N], dtype=np.complex)
            for n in range(N):
                self.separated_spec[:, :, n] = np.matmul( self.Y_FTN[:, :, n][:, :, None], np.linalg.inv(self.convert_to_NumpyArray( self.diagonalizer_FMM))[:, :, n][:, None] )[:, :, mic_index]
            self.separated_spec = self.separated_spec.transpose(2, 0, 1)
        else:
            self.separated_spec = np.matmul( self.Y_FTN[:, :, source_index][:, :, None], np.linalg.inv(self.convert_to_NumpyArray( self.diagonalizer_FMM))[:, :, source_index][:, None] )[:, :, mic_index]
        return self.separated_spec # N F T or F T


    def separate_GEV(self, mic_index=MIC_INDEX): # separate using GEV
        covarianceMatrix_NFMM = self.convert_to_NumpyArray(self.calculate_covarianceMatrix())
        self.separated_spec = separation.separate_GEV(X_FTM=self.convert_to_NumpyArray(self.X_FTM), target_covarianceMatrix_FMM=covarianceMatrix_NFMM[0], noise_covarianceMatrix_FMM=covarianceMatrix_NFMM[1:].sum(axis=0), mic_index=mic_index)


    def separate_MVDR(self, mic_index=MIC_INDEX): # separate using MVDR
        covarianceMatrix_NFMM = self.convert_to_NumpyArray(self.calculate_covarianceMatrix())
        self.separated_spec = separation.separate_MVDR_from_covarianceMatrix(X_FTM=self.convert_to_NumpyArray(self.X_FTM), covarianceMatrix_NFMM=covarianceMatrix_NFMM, mic_index=mic_index, source_index=0)


    def save_separated_signal(self, filename="sample.wav"):
        separation.save_separated_spec(self.convert_to_NumpyArray(self.separated_spec), save_filename=filename)


    def save_parameter(self, filename):
        param_list = [self.lambda_NFT, self.covarianceDiag_NFM, self.diagonalizer_FMM]
        if self.xp != np:
            param_list = [self.convert_to_NumpyArray(param) for param in param_list]
        pic.dump(param_list, open(filename, "wb"))


    def load_parameter(self, filename):
        param_list = pic.load(open(filename, "rb"))
        if self.xp != np:
            param_list = [cuda.to_gpu(param) for param in param_list]
        self.lambda_NFT, self.covarianceDiag_NFM, self.diagonalizer_FMM = param_list

        self.NUM_source, self.NUM_freq, self.NUM_time = self.lambda_NFT.shape
        self.NUM_mic = self.covarianceDiag_NFM.shape[-1]


    def calculate_separation_performance(self):
        if self.MODE_initialize_covarianceMatrix == "obs":
            separated_sig = librosa.core.istft(self.separated_spec[0])
        else:
            for n in range(self.NUM_source):
                tmp = librosa.core.istft(self.separated_spec[n])
                if n == 0:
                    separated_sig = np.zeros([self.NUM_source, len(tmp)])
                separated_sig[n] = tmp
        sdr = calculate_SDR_SIR_SAR(self.wav_org, separated_sig)[0][0]
        return sdr


if __name__ == "__main__":
    import argparse
    import pickle as pic
    import sys, os
    sys.path.append("/home/sekiguch/Dropbox/program/python/chainer/")
    sys.path.append("../chainer/")
    import separation

    parser = argparse.ArgumentParser()
    parser.add_argument(             '--gpu', type=  int, default=    0, help='GPU ID')##
    parser.add_argument(          '--n_freq', type=  int, default=  513, help='number of frequencies')
    parser.add_argument(       '--NUM_noise', type=  int, default=    1, help='number of noise')
    parser.add_argument(   '--NUM_iteration', type=  int, default=  100, help='number of iteration')
    parser.add_argument(       '--NUM_basis', type=  int, default=    8, help='number of basis')
    parser.add_argument( '--MODE_initialize_covarianceMatrix', type=  str, default="obs", help='cGMM, cGMM2, unit, obs')
    args = parser.parse_args()

    if args.gpu < 0:
        import numpy as xp
    else:
        import cupy as xp
        print("Use GPU " + str(args.gpu))
        cuda.get_device_from_id(args.gpu).use()

    wav, fs = sf.read("../../data/chime/F04_050C0115_CAF.CH13456.wav")
    wav = wav.T
    M = len(wav)
    for m in range(M):
        tmp = librosa.core.stft(wav[m], n_fft=1024, hop_length=256)
        if m == 0:
            spec = np.zeros([tmp.shape[0], tmp.shape[1], M], dtype=np.complex)
        spec[:, :, m] = tmp

    separater = FastFCA(NUM_source=args.NUM_noise+1, xp=xp, MODE_initialize_covarianceMatrix=args.MODE_initialize_covarianceMatrix)
    separater.load_spectrogram(spec)

    separater.file_id = "F04_050C0115_CAF"

    processingTime = separater.solve(NUM_iteration=args.NUM_iteration, save_likelihood=True, save_parameter=False, save_wav=False, save_dir="./", interval_save_parameter=10)
    print("processingTime : ", processingTime / args.NUM_iteration)
