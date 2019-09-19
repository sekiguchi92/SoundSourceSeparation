#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import chainer
import sys, os
from chainer import cuda, serializers
from chainer import functions as chf
from progressbar import progressbar
import librosa
import soundfile as sf
import pickle as pic

from FCA import FCA
from configure import *


class MNMF_DP(FCA):

    def __init__(self, speech_VAE=None, NUM_noise=1, NUM_Z_iteration=30, DIM_latent=16, NUM_basis_noise=2, xp=np, MODE_initialize_covarianceMatrix="unit", MODE_update_parameter=["all", "Z", "one_by_one"][1], MODE_update_Z=["sampling", "backprop"][0], normalize_encoder_input=True):
        """ initialize MNMF

        Parameters:
        -----------
            speech_VAE: VAE
                trained speech VAE network
            NUM_noise: int
                the number of noise sources
            NUM_Z_iteration: int
                the number of iteration for updating Z per global iteration
            DIM_latent: int
                the dimension of latent variable Z
            NUM_basis_noise: int
                the number of bases of each noise source
            xp : numpy or cupy
            MODE_initialize_covarianceMatrix: str
                how to initialize covariance matrix {unit, obs, ILRMA}
            MODE_update_parameter: str
                'all' : update all the variables simultanesouly
                'one_by_one' : update one by one
            MODE_update_Z: str
                how to update latent variable Z {sampling, backprop}
            normalize_encoder_input: boolean
                normalize observation to initialize latent variable by feeding the observation into a encoder
        """
        super(MNMF_DP, self).__init__(NUM_source=NUM_noise+1, xp=xp, MODE_initialize_covarianceMatrix=MODE_initialize_covarianceMatrix, MODE_update_parameter=MODE_update_parameter)
        self.NUM_source, self.NUM_noise, self.NUM_speech = NUM_noise+1, NUM_noise, 1
        self.NUM_basis_noise = NUM_basis_noise
        self.NUM_Z_iteration = NUM_Z_iteration
        self.DIM_latent = DIM_latent
        self.speech_VAE = speech_VAE
        self.MODE_update_Z = MODE_update_Z
        self.normalize_encoder_input = normalize_encoder_input
        self.method_name = "MNMF_DP"


    def set_parameter(self, NUM_noise=None, NUM_iteration=None, NUM_Z_iteration=None, NUM_basis_noise=None, MODE_initialize_covarianceMatrix=None, MODE_update_parameter=None, MODE_update_Z=None):
        """ set parameters

        Parameters:
        -----------
            NUM_noise: int
                the number of sources
            MODE_initialize_covarianceMatrix: str
                how to initialize covariance matrix {unit, obs, ILRMA}
            MODE_update_parameter: str
                'all' : update all the variables simultanesouly
                'Z' : update variables other than Z and then update Z
                'one_by_one' : update one by one
        """
        if NUM_noise != None:
            self.NUM_noise = NUM_noise
            self.NUM_source = NUM_noise + 1
        if NUM_iteration != None:
            self.NUM_iteration = NUM_iteration
        if NUM_Z_iteration != None:
            self.NUM_Z_iteration = NUM_Z_iteration
        if NUM_basis_noise != None:
            self.NUM_basis_noise = NUM_basis_noise
        if MODE_initialize_covarianceMatrix != None:
            self.MODE_initialize_covarianceMatrix = MODE_initialize_covarianceMatrix
        if MODE_update_parameter != None:
            self.MODE_update_parameter = MODE_update_parameter
        if MODE_update_Z != None:
            self.MODE_update_Z = MODE_update_Z


    def initialize_PSD(self):
        self.lambda_NFT = self.xp.zeros([self.NUM_source, self.NUM_freq, self.NUM_time]).astype(self.xp.float)
        self.power_speech_FT = self.xp.random.random([self.NUM_freq, self.NUM_time]).astype(self.xp.float)
        power_observation_FT = (self.xp.abs(self.X_FTM) ** 2).mean(axis=2)
        shape = 2
        self.W_noise_NnFK = self.xp.random.dirichlet(np.ones(self.NUM_freq)*shape, size=[self.NUM_noise, self.NUM_basis_noise]).transpose(0, 2, 1)
        self.H_noise_NnKT = self.xp.random.gamma(shape, (power_observation_FT.mean() * self.NUM_freq * self.NUM_mic / (self.NUM_noise * self.NUM_basis_noise)) / shape, size=[self.NUM_noise, self.NUM_basis_noise, self.NUM_time])
        self.H_noise_NnKT[self.H_noise_NnKT < EPS] = EPS

        self.u_F = self.xp.ones(self.NUM_freq) / self.NUM_freq
        self.v_T = self.xp.ones(self.NUM_time)

        if self.normalize_encoder_input:
            power_observation_FT = power_observation_FT / power_observation_FT.sum(axis=0).mean()
        self.Z_speech_DT = self.speech_VAE.encode_cupy(power_observation_FT.astype(self.xp.float32))
        self.z_link_speech = Z_link(self.Z_speech_DT.T)
        self.z_optimizer_speech = chainer.optimizers.Adam().setup(self.z_link_speech)

        self.power_speech_FT = self.speech_VAE.decode_cupy(self.Z_speech_DT)
        self.lambda_NFT[0] = self.u_F[:, None] * self.v_T[None] * self.power_speech_FT
        self.lambda_NFT[1:] = self.W_noise_NnFK @ self.H_noise_NnKT


    def make_filename_suffix(self):
        self.filename_suffix = "N={}-it={}-itZ={}-Kn={}-D={}-init={}-latent={}-update={}".format(self.NUM_noise, self.NUM_iteration, self.NUM_Z_iteration, self.NUM_basis_noise, self.DIM_latent, self.MODE_initialize_covarianceMatrix, self.MODE_update_Z, self.MODE_update_parameter)

        if hasattr(self, "name_DNN"):
           self.filename_suffix += "-DNN={}".format(self.name_DNN)
        else:
           self.filename_suffix += "-DNN=NoName"

        if hasattr(self, "file_id"):
           self.filename_suffix += "-ID={}".format(self.file_id)
        else:
            print("====================\n\nWarning: Please set self.file_id\n\n====================")

        print("filename_suffix:", self.filename_suffix)


    def update(self):
        if self.MODE_update_parameter == "one_by_one":
            self.update_axiliary_variable()
            self.update_W_noise()
            self.update_axiliary_variable()
            self.update_H_noise()
            self.update_axiliary_variable()
            self.update_covarianceMatrix()
            self.update_axiliary_variable()
            self.update_U()
            self.update_axiliary_variable()
            self.update_V()
            self.update_axiliary_variable()
            self.update_Z_speech(calc_constant=True)
            self.normalize()
        elif self.MODE_update_parameter == "all":
            self.update_axiliary_variable_and_Z()
            self.update_WH_noise()
            self.update_covarianceMatrix()
            self.update_UV()
            self.update_Z_speech(calc_constant=False)
            self.normalize()
        elif self.MODE_update_parameter == "Z":
            self.update_axiliary_variable_and_Z()
            self.update_WH_noise()
            self.update_covarianceMatrix()
            self.update_UV()
            self.update_Z_speech(calc_constant=True)
            self.normalize()


    def update_axiliary_variable_and_Z(self):
        Y_NFTMM = self.lambda_NFT[..., None, None] * self.covarianceMatrix_NFMM[:, :, None]
        if self.xp == np:
            self.Yinv_FTMM = np.linalg.inv(Y_NFTMM.sum(axis=0))
            Yx_FTM1 = self.Yinv_FTMM @ self.X_FTM[..., None]
            self.Yinv_X_Yinv_FTMM = Yx_FTM1 @ Yx_FTM1.conj().transpose(0, 1, 3, 2) # for reducing computational cost in case of CPU
            cov_inv_FMM = np.linalg.inv(self.covarianceMatrix_NFMM[0])
        else:
            self.Yinv_FTMM = self.calculateInverseMatrix(Y_NFTMM.sum(axis=0))
            Yx_FTM1 = self.Yinv_FTMM @ self.X_FTM[..., None]
            self.Yinv_X_Yinv_FTMM = Yx_FTM1 @ Yx_FTM1.conj().transpose(0, 1, 3, 2) # for reducing computational cost in case of CPU
            cov_inv_FMM = self.calculateInverseMatrix(self.covarianceMatrix_NFMM[0])

        self.tr_Cov_Yinv_X_Yinv_NFT = self.xp.trace(self.covarianceMatrix_NFMM[:, :, None] @ self.Yinv_X_Yinv_FTMM[None], axis1=3, axis2=4).real
        self.tr_Cov_Yinv_NFT = self.xp.trace(self.covarianceMatrix_NFMM[:, :, None] @ self.Yinv_FTMM[None], axis1=3, axis2=4).real

        Phi_FTMM = Y_NFTMM[0] @ self.Yinv_FTMM
        self.tr_Omega_Cov_FT = self.tr_Cov_Yinv_NFT[0]
        self.tr_Cov_Phi_X_Phi_FT = self.xp.trace(cov_inv_FMM[:, None] @ Phi_FTMM @ self.XX_FTMM @ Phi_FTMM.transpose(0, 1, 3, 2).conj(), axis1=2, axis2=3).real
        self.UV_FT = self.u_F[:, None] * self.v_T[None]


    def update_UV(self):
        a_1 = (self.u_F[:, None] * self.power_speech_FT * self.tr_Cov_Yinv_X_Yinv_NFT[0]).sum(axis=0)
        b_1 = (self.u_F[:, None] * self.power_speech_FT * self.tr_Cov_Yinv_NFT[0]).sum(axis=0)

        a_2 = (self.v_T[None] * self.power_speech_FT * self.tr_Cov_Yinv_X_Yinv_NFT[0]).sum(axis=1)
        b_2 = (self.v_T[None] * self.power_speech_FT * self.tr_Cov_Yinv_NFT[0]).sum(axis=1)

        self.v_T = self.v_T * self.xp.sqrt(a_1 / b_1)
        self.u_F = self.u_F * self.xp.sqrt(a_2 / b_2)


    def update_U(self):
        a_1 = (self.v_T[None] * self.power_speech_FT * self.tr_Cov_Yinv_X_Yinv_NFT[0]).sum(axis=1)
        b_1 = (self.v_T[None] * self.power_speech_FT * self.tr_Cov_Yinv_NFT[0]).sum(axis=1)
        self.u_F = self.u_F * self.xp.sqrt(a_1 / b_1)
        self.lambda_NFT[0] = self.u_F[:, None] * self.v_T[None] * self.power_speech_FT


    def update_V(self):
        a_1 = (self.u_F[:, None] * self.power_speech_FT * self.tr_Cov_Yinv_X_Yinv_NFT[0]).sum(axis=0)
        b_1 = (self.u_F[:, None] * self.power_speech_FT * self.tr_Cov_Yinv_NFT[0]).sum(axis=0)
        self.v_T = self.v_T * self.xp.sqrt(a_1 / b_1)
        self.UV_FT = self.u_F[:, None] * self.v_T[None]
        self.lambda_NFT[0] = self.u_F[:, None] * self.v_T[None] * self.power_speech_FT


    def update_WH_noise(self):
        if self.xp == np: # CPU
            a_2 = self.xp.zeros([self.NUM_noise, self.NUM_basis_noise, self.NUM_time])
            b_2 = self.xp.zeros([self.NUM_noise, self.NUM_basis_noise, self.NUM_time])

            for f in range(self.NUM_freq):
                a_1 = (self.H_noise_NnKT.transpose(0, 2, 1) * self.tr_Cov_Yinv_X_Yinv_NFT[1:, f, :, None]).sum(axis=1) # Nn K
                b_1 = (self.H_noise_NnKT.transpose(0, 2, 1) * self.tr_Cov_Yinv_NFT[1:, f, :, None]).sum(axis=1) # Nn K

                a_2 += (self.W_noise_NnFK[:, f, :, None] * self.tr_Cov_Yinv_X_Yinv_NFT[1:, f, None]) # Nn K T
                b_2 += (self.W_noise_NnFK[:, f, :, None] * self.tr_Cov_Yinv_NFT[1:, f, None]) # Nn K T
                self.W_noise_NnFK[:, f] = self.W_noise_NnFK[:, f] * self.xp.sqrt(a_1 / b_1)
            self.H_noise_NnKT = self.H_noise_NnKT * self.xp.sqrt(a_2 / b_2)
        else: # GPU
            a_1 = (self.H_noise_NnKT.transpose(0, 2, 1)[:, None] * self.tr_Cov_Yinv_X_Yinv_NFT[1:, :, :, None]).sum(axis=2) # Nn F K
            b_1 = (self.H_noise_NnKT.transpose(0, 2, 1)[:, None] * self.tr_Cov_Yinv_NFT[1:, :, :, None]).sum(axis=2) # Nn F K

            a_2 = (self.W_noise_NnFK[..., None] * self.tr_Cov_Yinv_X_Yinv_NFT[1:, :, None]).sum(axis=1) # Nn K T
            b_2 = (self.W_noise_NnFK[..., None] * self.tr_Cov_Yinv_NFT[1:, :, None]).sum(axis=1) # Nn K T

            self.W_noise_NnFK = self.W_noise_NnFK * self.xp.sqrt(a_1 / b_1)
            self.H_noise_NnKT = self.H_noise_NnKT * self.xp.sqrt(a_2 / b_2)


    def update_H_noise(self):
        if self.xp == np: # CPU
            a_1 = self.xp.zeros([self.NUM_noise, self.NUM_basis_noise, self.NUM_time])
            b_1 = self.xp.zeros([self.NUM_noise, self.NUM_basis_noise, self.NUM_time])
            for f in range(self.NUM_freq):
                a_1 += (self.W_noise_NnFK[:, f, :, None] * self.tr_Cov_Yinv_X_Yinv_NFT[1:, f, None]) # Nn K T
                b_1 += (self.W_noise_NnFK[:, f, :, None] * self.tr_Cov_Yinv_NFT[1:, f, None]) # Nn K T
            self.H_noise_NnKT = self.H_noise_NnKT * self.xp.sqrt(a_1 / b_1)
        else: # GPU
            a_1 = (self.W_noise_NnFK[..., None] * self.tr_Cov_Yinv_X_Yinv_NFT[1:, :, None]).sum(axis=1) # Nn K T
            b_1 = (self.W_noise_NnFK[..., None] * self.tr_Cov_Yinv_NFT[1:, :, None]).sum(axis=1) # Nn K T
            self.H_noise_NnKT = self.H_noise_NnKT * self.xp.sqrt(a_1 / b_1)
        self.lambda_NFT[1:] = self.W_noise_NnFK @ self.H_noise_NnKT


    def update_W_noise(self):
        if self.xp == np: # CPU
            for f in range(self.NUM_freq):
                a_1 = (self.H_noise_NnKT.transpose(0, 2, 1) * self.tr_Cov_Yinv_X_Yinv_NFT[1:, f, :, None]).sum(axis=1) # Nn K
                b_1 = (self.H_noise_NnKT.transpose(0, 2, 1) * self.tr_Cov_Yinv_NFT[1:, f, :, None]).sum(axis=1) # Nn K
                self.W_noise_NnFK[:, f] = self.W_noise_NnFK[:, f] * self.xp.sqrt(a_1 / b_1)
        else: # GPU
            a_1 = (self.H_noise_NnKT.transpose(0, 2, 1)[:, None] * self.tr_Cov_Yinv_X_Yinv_NFT[1:, :, :, None]).sum(axis=2) # Nn F K
            b_1 = (self.H_noise_NnKT.transpose(0, 2, 1)[:, None] * self.tr_Cov_Yinv_NFT[1:, :, :, None]).sum(axis=2) # Nn F K
            self.W_noise_NnFK = self.W_noise_NnFK * self.xp.sqrt(a_1 / b_1)
        self.lambda_NFT[1:] = self.W_noise_NnFK @ self.H_noise_NnKT


    def normalize(self):
        mu_NF = self.xp.trace(self.covarianceMatrix_NFMM, axis1=2, axis2=3).real
        self.covarianceMatrix_NFMM = self.covarianceMatrix_NFMM / mu_NF[:, :, None, None]
        self.u_F = self.u_F * mu_NF[0]
        self.W_noise_NnFK = self.W_noise_NnFK * mu_NF[1:][:, :, None]

        nu = self.u_F.sum()
        self.u_F = self.u_F / nu
        self.v_T = nu * self.v_T

        nu_NnK = self.W_noise_NnFK.sum(axis=1)
        self.W_noise_NnFK = self.W_noise_NnFK / nu_NnK[:, None]
        self.H_noise_NnKT = self.H_noise_NnKT * nu_NnK[:, :, None]

        self.lambda_NFT[0] = self.u_F[:, None] * self.v_T[None] * self.power_speech_FT
        self.lambda_NFT[1:] = self.W_noise_NnFK @ self.H_noise_NnKT


    def loss_func_Z(self, z, vae, n):
        power_FT = chf.exp(vae.decode(z).T) * self.UV_FT + EPS
        if n == 0:
            loss = chf.sum(1 / power_FT * self.tr_Cov_Phi_X_Phi_FT + power_FT * self.tr_Omega_Cov_FT)
        else:
            raise NotImplementedError
        return loss


    def update_Z_speech(self, var_propose_distribution=1e-4, calc_constant=True):
        """
        Parameters:
            var_propose_distribution: float
                the variance of the propose distribution

        Results:
            self.Z_speech_DT: self.xp.array [ DIM_latent x T ]
                the latent variable of each speech
        """
        if calc_constant:
            self.calculate_constant_for_update_Z()

        if "backprop" in self.MODE_update_Z: # acceptance rate is calculated from likelihood
            for it in range(self.NUM_Z_iteration):
                with chainer.using_config('train', False):
                    self.z_optimizer_speech.update(self.loss_func_Z, self.z_link_speech.z, self.speech_VAE, 0)

            self.Z_speech_DT = self.z_link_speech.z.data.T
            self.power_speech_FT = self.speech_VAE.decode_cupy(self.Z_speech_DT)

        if "sampling" in self.MODE_update_Z:
            log_var = self.xp.log(self.xp.ones_like(self.Z_speech_DT).astype(self.xp.float32) * var_propose_distribution)
            Z_speech_old_DT = self.Z_speech_DT
            lambda_speech_old_FT = self.speech_VAE.decode_cupy(Z_speech_old_DT) * self.UV_FT
            for it in range(self.NUM_Z_iteration):
                Z_speech_new_DT = chf.gaussian(Z_speech_old_DT, log_var).data
                lambda_speech_new_FT = self.speech_VAE.decode_cupy(Z_speech_new_DT) * self.UV_FT
                acceptance_rate =  self.xp.exp((-1 * (1/lambda_speech_new_FT - 1/lambda_speech_old_FT) * self.tr_Cov_Phi_X_Phi_FT -  (lambda_speech_new_FT - lambda_speech_old_FT) * self.tr_Omega_Cov_FT).sum(axis=0) - (Z_speech_new_DT ** 2 - Z_speech_old_DT ** 2).sum(axis=0)/2)
                acceptance_boolean = self.xp.random.random([self.NUM_time]) < acceptance_rate
                Z_speech_old_DT[:, acceptance_boolean] = Z_speech_new_DT[:, acceptance_boolean]
                lambda_speech_old_FT[:, acceptance_boolean] = lambda_speech_new_FT[:, acceptance_boolean]

            self.Z_speech_DT = Z_speech_old_DT
            self.z_link_speech.z = chainer.Parameter(self.Z_speech_DT.T)
            self.power_speech_FT = self.speech_VAE.decode_cupy(self.Z_speech_DT)


    def calculate_constant_for_update_Z(self):
        Y_NFTMM = self.lambda_NFT[..., None, None] * self.covarianceMatrix_NFMM[:, :, None]
        if self.xp == np:
            self.Yinv_FTMM = np.linalg.inv(Y_NFTMM.sum(axis=0))
            cov_inv_FMM = np.linalg.inv(self.covarianceMatrix_NFMM[0])
        else:
            self.Yinv_FTMM = self.calculateInverseMatrix(Y_NFTMM.sum(axis=0))
            cov_inv_FMM = self.calculateInverseMatrix(self.covarianceMatrix_NFMM[0])

        Phi_FTMM = Y_NFTMM[0] @ self.Yinv_FTMM
        self.tr_Omega_Cov_FT = self.xp.trace(self.covarianceMatrix_NFMM[0, :, None] @ self.Yinv_FTMM, axis1=2, axis2=3).real
        self.tr_Cov_Phi_X_Phi_FT = self.xp.trace(cov_inv_FMM[:, None] @ Phi_FTMM @ self.XX_FTMM @ Phi_FTMM.transpose(0, 1, 3, 2).conj(), axis1=2, axis2=3).real


    def save_parameter(self, filename):
        param_list = [self.covarianceMatrix_NFMM, self.lambda_NFT]
        param_list.extend([self.u_F, self.v_T, self.Z_speech_DT])
        param_list.append([self.W_noise_NnFK, self.H_noise_NnKT])

        if self.xp != np:
            param_list = [cuda.to_cpu(param) for param in param_list]

        pic.dump(param_list, open(filename, "wb"))


    def load_parameter(self, filename):
        param_list = pic.load(open(filename, "rb"))
        if self.xp != np:
            param_list = [cuda.to_gpu(param) for param in param_list]

        self.covarianceMatrix_NFMM, self.lambda_NFT, self.u_F, self.v_T, self.Z_speech_DT, self.W_noise_NnFK, self.H_noise_NnKT = param_list
        self.NUM_source, self.NUM_freq, self.NUM_time = self.lambda_NFT.shape
        self.NUM_mic = self.covarianceMatrix_NFMM.shape[-1]
        self.DIM_latent = self.Z_speech_DT.shape[0]
        self.NUM_noise, self.NUM_speech = self.NUM_source - 1, 1



class Z_link(chainer.link.Link):
    def __init__(self, z):
        super(Z_link, self).__init__()

        with self.init_scope():
            self.z = chainer.Parameter(z)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(        'input_fileName', type= str, help='filename of the multichannel observed signals')
    parser.add_argument(             '--file_id', type= str, default="None", help='file id')
    parser.add_argument(                 '--gpu', type= int, default=     0, help='GPU ID')##
    parser.add_argument(               '--n_fft', type= int, default=  1024, help='number of frequencies')
    parser.add_argument(           '--NUM_noise', type= int, default=     1, help='number of noise')
    parser.add_argument(          '--DIM_latent', type= int, default=    16, help='dimention of encoded vector')
    parser.add_argument(       '--MODE_update_Z', type= str, default="sampling", help='sampling, sampling2, backprop, backprop2, hybrid, hybrid2')
    parser.add_argument(       '--NUM_iteration', type= int, default=    30, help='number of iteration')
    parser.add_argument(     '--NUM_Z_iteration', type= int, default=    30, help='number of update Z iteration')
    parser.add_argument(     '--NUM_basis_noise', type= int, default=    64, help='number of basis of noise (MODE_noise=NMF)')
    parser.add_argument('--MODE_update_parameter', type= str, default= "all", help='all, one_by_one')
    parser.add_argument('--MODE_initialize_covarianceMatrix', type=  str, default="obs", help='unit, obs, ILRMA')
    args = parser.parse_args()


    sys.path.append("../DeepSpeechPrior")
    import network_VAE
    model_fileName = "../DeepSpeechPrior/model-VAE-best-scale=gamma-D={}.npz".format(args.DIM_latent)
    speech_VAE = network_VAE.VAE(n_latent=args.DIM_latent)
    serializers.load_npz(model_fileName, speech_VAE)
    name_DNN = "VAE"

    if args.gpu < 0:
        import numpy as xp
    else:
        import cupy as xp
        print("Use GPU " + str(args.gpu))
        cuda.get_device_from_id(args.gpu).use()
        speech_VAE.to_gpu()

    wav, fs = sf.read(args.input_fileName)
    wav = wav.T
    M = len(wav)
    for m in range(M):
        tmp = librosa.core.stft(wav[m], n_fft=args.n_fft, hop_length=int(args.n_fft/4))
        if m == 0:
            spec = np.zeros([tmp.shape[0], tmp.shape[1], M], dtype=np.complex)
        spec[:, :, m] = tmp

    separater = MNMF_DP(NUM_noise=args.NUM_noise, NUM_Z_iteration=args.NUM_Z_iteration, speech_VAE=speech_VAE, DIM_latent=args.DIM_latent, NUM_basis_noise=args.NUM_basis_noise, xp=xp, MODE_initialize_covarianceMatrix=args.MODE_initialize_covarianceMatrix, MODE_update_parameter=args.MODE_update_parameter)

    separater.load_spectrogram(spec)
    separater.name_DNN = name_DNN
    separater.file_id = args.file_id
    separater.solve(NUM_iteration=args.NUM_iteration, save_likelihood=False, save_parameter=False, save_path="./", interval_save_parameter=100)
