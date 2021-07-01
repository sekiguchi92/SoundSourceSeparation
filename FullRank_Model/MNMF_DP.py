#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import chainer
import sys, os
from chainer import cuda, serializers
from chainer import functions as chf
import librosa
import soundfile as sf
import pickle as pic

from FCA import FCA
from configure import *


class MNMF_DP(FCA):
    """ Blind Speech Enhancement Using Multichannel Nonnegative Matrix Factorization with a Deep Speech Prior (MNMF-DP)

    X_FTM: the observed complex spectrogram
    covarianceMatrix_NFMM: spatial covariance matrices (SCMs) for each source
    W_noise_NnFK: basis vectors for noise sources (Nn means the number of noise sources)
    H_noise_NnKT: activations for noise sources
    Z_speech_DT: latent variables for speech
    power_speech_FT: power spectra of speech that is the output of DNN(Z_speech_DT)
    lambda_NFT: power spectral densities of each source
        lambda_NFT[0] = U_F * V_T * power_speech_FT
        lambda_NFT[1:] = W_noise_NnFK @ H_noise_NnKT
    """

    def __init__(self, speech_VAE=None, n_noise=1, n_Z_iteration=30, n_latent=16, n_basis_noise=2, xp=np, init_SCM="unit", mode_update_parameter=["all", "Z", "one_by_one"][1],\
            mode_update_Z=["sampling", "backprop"][0], normalize_encoder_input=True, seed=0):
        """ initialize MNMF_DP

        Parameters:
        -----------
            speech_VAE: VAE
                trained speech VAE network
            n_noise: int
                the number of noise sources
            n_Z_iteration: int
                the number of iteration for updating Z per global iteration
            n_latent: int
                the dimension of latent variable Z
            n_basis_noise: int
                the number of bases of each noise source
            xp : numpy or cupy
            init_SCM: str
                how to initialize covariance matrix {unit, obs, ILRMA}
            mode_update_parameter: str
                'all' : update all the variables simultanesouly
                'one_by_one' : update one by one
            mode_update_Z: str
                how to update latent variable Z {sampling, backprop}
            normalize_encoder_input: boolean
                normalize observation to initialize latent variable by feeding the observation into a encoder
        """
        super(MNMF_DP, self).__init__(n_source=n_noise+1, xp=xp, init_SCM=init_SCM, mode_update_parameter=mode_update_parameter, seed=seed)
        self.method_name = "MNMF_DP"
        self.n_source, self.n_noise, self.n_speech = n_noise+1, n_noise, 1
        self.n_basis_noise = n_basis_noise
        self.n_Z_iteration = n_Z_iteration
        self.n_latent = n_latent
        self.speech_VAE = speech_VAE
        self.mode_update_Z = mode_update_Z
        self.normalize_encoder_input = normalize_encoder_input


    def initialize_PSD(self):
        self.W_noise_NnFK = self.xp.random.rand(self.n_noise, self.n_freq, self.n_basis_noise).astype(self.xp.float)
        self.H_noise_NnKT = self.xp.random.rand(self.n_noise, self.n_basis_noise, self.n_time).astype(self.xp.float)

        self.u_F = self.xp.ones(self.n_freq, dtype=self.xp.float) / self.n_freq
        self.v_T = self.xp.ones(self.n_time, dtype=self.xp.float)

        power_observation_FT = (self.xp.abs(self.X_FTM) ** 2).mean(axis=2)
        if self.normalize_encoder_input:
            power_observation_FT = power_observation_FT / power_observation_FT.sum(axis=0).mean()
        self.Z_speech_DT = self.speech_VAE.encode_cupy(power_observation_FT.astype(self.xp.float32))
        self.z_link_speech = Z_link(self.Z_speech_DT.T)
        self.z_optimizer_speech = chainer.optimizers.Adam().setup(self.z_link_speech)
        self.power_speech_FT = self.speech_VAE.decode_cupy(self.Z_speech_DT)

        self.lambda_NFT = self.xp.zeros([self.n_source, self.n_freq, self.n_time]).astype(self.xp.float)
        self.lambda_NFT[0] = self.u_F[:, None] * self.v_T[None] * self.power_speech_FT
        self.lambda_NFT[1:] = self.W_noise_NnFK @ self.H_noise_NnKT


    def make_filename_suffix(self):
        self.filename_suffix = f"N={self.n_noise}-it={self.n_iteration}-itZ={self.n_Z_iteration}-Kn={self.n_basis_noise}-D={self.n_latent}-init={self.init_SCM}-latent={self.mode_update_Z}-update={self.mode_update_parameter}"

        if hasattr(self, "name_DNN"):
            self.filename_suffix += f"-DNN={self.name_DNN}"
        if hasattr(self, "file_id"):
            self.filename_suffix += f"-ID={self.file_id}"
        print("param:", self.filename_suffix)


    def update(self):
        if self.mode_update_parameter == "one_by_one":
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
        elif self.mode_update_parameter == "all":
            self.update_axiliary_variable_and_Z()
            self.update_WH_noise()
            self.update_covarianceMatrix()
            self.update_UV()
            self.update_Z_speech(calc_constant=False)
            self.normalize()
        elif self.mode_update_parameter == "Z":
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
            self.Yinv_FTMM = self.xp.linalg.inv(Y_NFTMM.sum(axis=0))
            Yx_FTM1 = self.Yinv_FTMM @ self.X_FTM[..., None]
            self.Yinv_X_Yinv_FTMM = Yx_FTM1 @ Yx_FTM1.conj().transpose(0, 1, 3, 2) # for reducing computational cost in case of CPU
            cov_inv_FMM = self.xp.linalg.inv(self.covarianceMatrix_NFMM[0])

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

        self.v_T *= self.xp.sqrt(a_1 / b_1)
        self.u_F *= self.xp.sqrt(a_2 / b_2)


    def update_U(self):
        a_1 = (self.v_T[None] * self.power_speech_FT * self.tr_Cov_Yinv_X_Yinv_NFT[0]).sum(axis=1)
        b_1 = (self.v_T[None] * self.power_speech_FT * self.tr_Cov_Yinv_NFT[0]).sum(axis=1)
        self.u_F *= self.xp.sqrt(a_1 / b_1)
        self.lambda_NFT[0] = self.u_F[:, None] * self.v_T[None] * self.power_speech_FT


    def update_V(self):
        a_1 = (self.u_F[:, None] * self.power_speech_FT * self.tr_Cov_Yinv_X_Yinv_NFT[0]).sum(axis=0)
        b_1 = (self.u_F[:, None] * self.power_speech_FT * self.tr_Cov_Yinv_NFT[0]).sum(axis=0)
        self.v_T *= self.xp.sqrt(a_1 / b_1)
        self.UV_FT = self.u_F[:, None] * self.v_T[None]
        self.lambda_NFT[0] = self.u_F[:, None] * self.v_T[None] * self.power_speech_FT


    def update_WH_noise(self):
        a_1 = (self.H_noise_NnKT.transpose(0, 2, 1)[:, None] * self.tr_Cov_Yinv_X_Yinv_NFT[1:, :, :, None]).sum(axis=2) # Nn F K
        b_1 = (self.H_noise_NnKT.transpose(0, 2, 1)[:, None] * self.tr_Cov_Yinv_NFT[1:, :, :, None]).sum(axis=2) # Nn F K

        a_2 = (self.W_noise_NnFK[..., None] * self.tr_Cov_Yinv_X_Yinv_NFT[1:, :, None]).sum(axis=1) # Nn K T
        b_2 = (self.W_noise_NnFK[..., None] * self.tr_Cov_Yinv_NFT[1:, :, None]).sum(axis=1) # Nn K T

        self.W_noise_NnFK *= self.xp.sqrt(a_1 / b_1)
        self.H_noise_NnKT *= self.xp.sqrt(a_2 / b_2)


    def update_H_noise(self):
        a_1 = (self.W_noise_NnFK[..., None] * self.tr_Cov_Yinv_X_Yinv_NFT[1:, :, None]).sum(axis=1) # Nn K T
        b_1 = (self.W_noise_NnFK[..., None] * self.tr_Cov_Yinv_NFT[1:, :, None]).sum(axis=1) # Nn K T
        self.H_noise_NnKT *= self.xp.sqrt(a_1 / b_1)
        self.lambda_NFT[1:] = self.W_noise_NnFK @ self.H_noise_NnKT + EPS


    def update_W_noise(self):
        a_1 = (self.H_noise_NnKT.transpose(0, 2, 1)[:, None] * self.tr_Cov_Yinv_X_Yinv_NFT[1:, :, :, None]).sum(axis=2) # Nn F K
        b_1 = (self.H_noise_NnKT.transpose(0, 2, 1)[:, None] * self.tr_Cov_Yinv_NFT[1:, :, :, None]).sum(axis=2) # Nn F K
        self.W_noise_NnFK *= self.xp.sqrt(a_1 / b_1)
        self.lambda_NFT[1:] = self.W_noise_NnFK @ self.H_noise_NnKT + EPS


    def normalize(self):
        mu_NF = self.xp.trace(self.covarianceMatrix_NFMM, axis1=2, axis2=3).real
        self.covarianceMatrix_NFMM = self.covarianceMatrix_NFMM / mu_NF[:, :, None, None]
        self.u_F *= mu_NF[0]
        self.W_noise_NnFK *= mu_NF[1:][:, :, None]

        nu = self.u_F.sum()
        self.u_F /= nu
        self.v_T *= nu

        nu_NnK = self.W_noise_NnFK.sum(axis=1)
        self.W_noise_NnFK /= nu_NnK[:, None]
        self.H_noise_NnKT *= nu_NnK[:, :, None]

        self.lambda_NFT[0] = self.u_F[:, None] * self.v_T[None] * self.power_speech_FT
        self.lambda_NFT[1:] = self.W_noise_NnFK @ self.H_noise_NnKT + EPS


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
            self.Z_speech_DT: self.xp.array [ n_latent x T ]
                the latent variable of each speech
        """
        if calc_constant:
            self.calculate_constant_for_update_Z()

        if "backprop" in self.mode_update_Z: # acceptance rate is calculated from likelihood
            for it in range(self.n_Z_iteration):
                with chainer.using_config('train', False):
                    self.z_optimizer_speech.update(self.loss_func_Z, self.z_link_speech.z, self.speech_VAE, 0)

            self.Z_speech_DT = self.z_link_speech.z.data.T
            self.power_speech_FT = self.speech_VAE.decode_cupy(self.Z_speech_DT)

        if "sampling" in self.mode_update_Z:
            log_var = self.xp.log(self.xp.ones_like(self.Z_speech_DT).astype(self.xp.float32) * var_propose_distribution)
            Z_speech_old_DT = self.Z_speech_DT
            lambda_speech_old_FT = self.speech_VAE.decode_cupy(Z_speech_old_DT) * self.UV_FT
            for it in range(self.n_Z_iteration):
                Z_speech_new_DT = chf.gaussian(Z_speech_old_DT, log_var).data
                lambda_speech_new_FT = self.speech_VAE.decode_cupy(Z_speech_new_DT) * self.UV_FT
                acceptance_rate =  self.xp.exp((-1 * (1/lambda_speech_new_FT - 1/lambda_speech_old_FT) * self.tr_Cov_Phi_X_Phi_FT -  (lambda_speech_new_FT - lambda_speech_old_FT) * self.tr_Omega_Cov_FT).sum(axis=0) - (Z_speech_new_DT ** 2 - Z_speech_old_DT ** 2).sum(axis=0)/2)
                acceptance_boolean = self.xp.random.random([self.n_time]) < acceptance_rate
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
            self.Yinv_FTMM = self.xp.linalg.inv(Y_NFTMM.sum(axis=0))
            cov_inv_FMM = self.xp.linalg.inv(self.covarianceMatrix_NFMM[0])

        Phi_FTMM = Y_NFTMM[0] @ self.Yinv_FTMM
        self.tr_Omega_Cov_FT = self.xp.trace(self.covarianceMatrix_NFMM[0, :, None] @ self.Yinv_FTMM, axis1=2, axis2=3).real
        self.tr_Cov_Phi_X_Phi_FT = self.xp.trace(cov_inv_FMM[:, None] @ Phi_FTMM @ self.XX_FTMM @ Phi_FTMM.transpose(0, 1, 3, 2).conj(), axis1=2, axis2=3).real


    def save_parameter(self, filename):
        param_list = [self.covarianceMatrix_NFMM, self.lambda_NFT, self.u_F, self.v_T, self.Z_speech_DT, self.W_noise_NnFK, self.H_noise_NnKT]

        if self.xp != np:
            param_list = [cuda.to_cpu(param) for param in param_list]

        pic.dump(param_list, open(filename, "wb"))


    def load_parameter(self, filename):
        param_list = pic.load(open(filename, "rb"))
        if self.xp != np:
            param_list = [cuda.to_gpu(param) for param in param_list]

        self.covarianceMatrix_NFMM, self.lambda_NFT, self.u_F, self.v_T, self.Z_speech_DT, self.W_noise_NnFK, self.H_noise_NnKT = param_list
        self.n_source, self.n_freq, self.n_time = self.lambda_NFT.shape
        self.n_mic = self.covarianceMatrix_NFMM.shape[-1]
        self.n_latent = self.Z_speech_DT.shape[0]
        self.n_noise, self.n_speech = self.n_source - 1, 1



class Z_link(chainer.link.Link):
    def __init__(self, z):
        super(Z_link, self).__init__()

        with self.init_scope():
            self.z = chainer.Parameter(z)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(         'input_fileName', type= str, help='filename of the multichannel observed signals')
    parser.add_argument(              '--file_id', type= str, default="None", help='file id')
    parser.add_argument(                  '--gpu', type= int, default=     0, help='GPU ID')##
    parser.add_argument(                '--n_fft', type= int, default=  1024, help='number of frequencies')
    parser.add_argument(              '--n_noise', type= int, default=     1, help='number of noise')
    parser.add_argument(             '--n_latent', type= int, default=    16, help='dimention of encoded vector')
    parser.add_argument(                '--n_mic', type= int, default=     8, help='number of microphones')
    parser.add_argument(        '--n_basis_noise', type= int, default=    64, help='number of basis of noise (MODE_noise=NMF)')
    parser.add_argument(             '--init_SCM', type=  str, default="obs", help='unit, obs, ILRMA')
    parser.add_argument(          '--n_iteration', type= int, default=    30, help='number of iteration')
    parser.add_argument(        '--n_Z_iteration', type= int, default=    30, help='number of update Z iteration')
    parser.add_argument(        '--mode_update_Z', type= str, default="sampling", help='sampling, sampling2, backprop, backprop2, hybrid, hybrid2')
    parser.add_argument('--mode_update_parameter', type= str, default= "all", help='all, one_by_one')
    args = parser.parse_args()


    sys.path.append("../DeepSpeechPrior")
    import network_VAE
    model_fileName = f"../DeepSpeechPrior/model-VAE-best-scale=gamma-D={args.n_latent}.npz"
    speech_VAE = network_VAE.VAE(n_latent=args.n_latent)
    serializers.load_npz(model_fileName, speech_VAE)
    name_DNN = "VAE"

    if args.gpu < 0:
        import numpy as xp
    else:
        import cupy as xp
        print("Use GPU " + str(args.gpu))
        xp.cuda.Device(args.gpu).use()
        speech_VAE.to_gpu()

    wav, fs = sf.read(args.input_fileName)
    wav = wav.T
    M = min(args.n_mic, len(wav))
    for m in range(M):
        tmp = librosa.core.stft(wav[m], n_fft=args.n_fft, hop_length=int(args.n_fft/4))
        if m == 0:
            spec = np.zeros([tmp.shape[0], tmp.shape[1], M], dtype=np.complex)
        spec[:, :, m] = tmp

    separater = MNMF_DP(n_noise=args.n_noise, n_Z_iteration=args.n_Z_iteration, speech_VAE=speech_VAE, n_latent=args.n_latent, n_basis_noise=args.n_basis_noise, xp=xp, init_SCM=args.init_SCM, mode_update_parameter=args.mode_update_parameter, seed=0)

    separater.load_spectrogram(spec)
    separater.name_DNN = name_DNN
    separater.file_id = args.file_id
    separater.solve(n_iteration=args.n_iteration, save_likelihood=False, save_parameter=False, save_path="./", interval_save_parameter=100)
