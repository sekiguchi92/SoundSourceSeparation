#! /usr/bin/env python3
# coding: utf-8

import sys, os
import numpy as np
import chainer
from chainer import functions as chf
import pickle as pic

from configure_FastModel import *
from FastFCA import FastFCA

try:
    from chainer import cuda
    FLAG_GPU_Available = True
except:
    print("---Warning--- You cannot use GPU acceleration because chainer or cupy is not installed")

class FastMNMF_DP(FastFCA):

    def __init__(self, speech_VAE=None, NUM_noise=1, NUM_Z_iteration=30, DIM_latent=16, NUM_basis_noise=2, xp=np, MODE_initialize_covarianceMatrix="unit", MODE_update_Z="sampling", normalize_encoder_input=True):
        """ initialize FastMNMF_DP

        Parameters:
        -----------
            NUM_noise: int
                the number of noise sources
            speech_VAE: VAE
                trained speech VAE network (necessary if you use VAE as speech model)
            DIM_latent: int
                the dimension of latent variable Z
            NUM_basis_noise: int
                the number of bases of each noise source
            MODE_initialize_covarianceMatrix: str
                how to initialize covariance matrix {unit, obs, cGMM, cGMM2(only speech)}
            MODE_update_Z: str
                how to update latent variable Z {sampling, backprop}
        """
        super(FastMNMF_DP, self).__init__(NUM_source=NUM_noise+1, xp=xp, MODE_initialize_covarianceMatrix=MODE_initialize_covarianceMatrix)
        self.NUM_source, self.NUM_speech, self.NUM_noise = NUM_noise+1, 1, NUM_noise
        self.speech_VAE = speech_VAE
        self.NUM_Z_iteration = NUM_Z_iteration
        self.NUM_basis_noise = NUM_basis_noise
        self.DIM_latent = DIM_latent
        self.MODE_update_Z = MODE_update_Z
        self.normalize_encoder_input = normalize_encoder_input
        self.method_name = "FastMNMF_DP"


    def load_spectrogram(self, X_FTM):
        """ load complex spectrogram
        Parameters:
        -----------
        X_FTM: xp.array [F x T x M]
        """
        self.xp = self.speech_VAE.xp
        super(FastMNMF_DP, self).load_spectrogram(X_FTM)
        self.u_F = self.xp.random.rand(self.NUM_freq).astype(self.xp.float)
        self.v_T = (self.xp.random.rand(self.NUM_time).astype(self.xp.float) * 0.9) + 0.1
        self.Z_speech_DT = self.xp.random.normal(0, 1, [self.DIM_latent, self.NUM_time]).astype(self.xp.float32)
        self.z_link_speech = Z_link(self.Z_speech_DT.T)
        self.z_optimizer_speech = chainer.optimizers.Adam().setup(self.z_link_speech)

        self.W_noise_NnFK = self.xp.abs(self.xp.random.rand(self.NUM_noise, self.NUM_freq, self.NUM_basis_noise).astype(self.xp.float))
        self.H_noise_NnKT = self.xp.abs(self.xp.random.rand(self.NUM_noise, self.NUM_basis_noise, self.NUM_time).astype(self.xp.float))

        self.power_speech_FT = self.speech_VAE.decode_cupy(self.Z_speech_DT)
        self.lambda_NFT[0] = self.u_F[:, None] * self.v_T[None] * self.power_speech_FT
        self.lambda_NFT[1:] = self.W_noise_NnFK @ self.H_noise_NnKT


    def set_parameter(self, NUM_noise=None, NUM_iteration=None, NUM_Z_iteration=None, NUM_basis_noise=None, MODE_initialize_covarianceMatrix=None, MODE_update_Z=None):
        """ set parameters

        Parameters:
        -----------
            NUM_noise: int
                the number of sources
            NUM_iteration: int
                the number of iteration
            NUM_Z_iteration: int
                the number of iteration of updating Z in each iteration
            NUM_basis_noise: int
                the number of basis of noise sources
            MODE_initialize_covarianceMatrix: str
                how to initialize covariance matrix {unit, obs, cGMM}
            MODE_update_Z: str
                how to update latent variable Z {sampling, backprop}
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
        if MODE_update_Z != None:
            self.MODE_update_Z = MODE_update_Z


    def initialize_PSD(self):
        """ 
        initialize parameters related to power spectral density (PSD)
        W, H, U, V, Z
        """
        power_observation_FT = (self.xp.abs(self.X_FTM) ** 2).mean(axis=2)
        shape = 2
        self.W_noise_NnFK = self.xp.random.dirichlet(np.ones(self.NUM_freq)*shape, size=[self.NUM_noise, self.NUM_basis_noise]).transpose(0, 2, 1)

        self.H_noise_NnKT[:] = self.xp.random.gamma(shape, (power_observation_FT.mean() * self.NUM_freq * self.NUM_mic / (self.NUM_noise * self.NUM_basis_noise)) / shape, size=[self.NUM_noise, self.NUM_basis_noise, self.NUM_time])
        self.H_noise_NnKT[self.H_noise_NnKT < EPS] = EPS

        self.u_F[:] = 1 / self.NUM_freq
        self.v_T[:] = 1

        if self.normalize_encoder_input:
            power_observation_FT = power_observation_FT / power_observation_FT.sum(axis=0).mean()

        self.Z_speech_DT = self.speech_VAE.encode_cupy(power_observation_FT.astype(self.xp.float32))
        self.z_link_speech = Z_link(self.Z_speech_DT.T)
        self.z_optimizer_speech = chainer.optimizers.Adam().setup(self.z_link_speech)
        self.power_speech_FT = self.speech_VAE.decode_cupy(self.Z_speech_DT)

        self.lambda_NFT[0] = self.u_F[:, None] * self.v_T[None] * self.power_speech_FT
        self.lambda_NFT[1:] = self.W_noise_NnFK @ self.H_noise_NnKT
        self.reset_variable()


    def make_fileName_suffix(self):
        self.fileName_suffix = "N={}-it={}-itZ={}-Ln={}-D={}-init={}-latent={}".format(self.NUM_noise, self.NUM_iteration, self.NUM_Z_iteration, self.NUM_basis_noise, self.DIM_latent, self.MODE_initialize_covarianceMatrix, self.MODE_update_Z)

        if hasattr(self, "name_DNN"):
            self.fileName_suffix += "-DNN={}".format(self.name_DNN)

        if hasattr(self, "file_id"):
            self.fileName_suffix += "-ID={}".format(self.file_id)
        else:
            print("====================\n\nWarning: Please set self.file_id\n\n====================")

        print("parameter:", self.fileName_suffix)


    def update(self):
        self.update_UV()
        self.update_Z_speech()
        self.update_WH_noise()
        self.update_CovarianceDiagElement()
        self.udpate_Diagonalizer()
        self.normalize()


    def normalize(self):
        phi_F = self.xp.sum(self.diagonalizer_FMM * self.diagonalizer_FMM.conj(), axis=(1, 2)).real / self.NUM_mic
        self.diagonalizer_FMM = self.diagonalizer_FMM / self.xp.sqrt(phi_F)[:, None, None]
        self.covarianceDiag_NFM = self.covarianceDiag_NFM / phi_F[None, :, None]

        mu_NF = (self.covarianceDiag_NFM).sum(axis=2).real
        self.covarianceDiag_NFM = self.covarianceDiag_NFM / mu_NF[:, :, None]
        self.u_F = self.u_F * mu_NF[0]
        self.W_noise_NnFK = self.W_noise_NnFK * mu_NF[1:][:, :, None]

        nu = self.u_F.sum()
        self.u_F = self.u_F / nu
        self.v_T = nu * self.v_T
        self.lambda_NFT[0] = self.u_F[:, None] * self.v_T[None] * self.power_speech_FT

        nu_NnK = self.W_noise_NnFK.sum(axis=1)
        self.W_noise_NnFK = self.W_noise_NnFK / nu_NnK[:, None]
        self.H_noise_NnKT = self.H_noise_NnKT * nu_NnK[:, :, None]
        self.lambda_NFT[1:] = self.W_noise_NnFK @ self.H_noise_NnKT

        self.reset_variable()


    def update_WH_noise(self):
        tmp1_NnFT = (self.covarianceDiag_NFM[1, :, None] * (self.Qx_power_FTM / (self.Y_FTM ** 2))[None]).sum(axis=3)
        tmp2_NnFT = (self.covarianceDiag_NFM[1, :, None] / self.Y_FTM[None]).sum(axis=3)
        a_W = (self.H_noise_NnKT[:, None] * tmp1_NnFT[:, :, None]).sum(axis=3)  # N F K T M
        b_W = (self.H_noise_NnKT[:, None] * tmp2_NnFT[:, :, None]).sum(axis=3)
        a_H = (self.W_noise_NnFK[..., None] * tmp1_NnFT[:, :, None] ).sum(axis=1) # N F K T M
        b_H = (self.W_noise_NnFK[..., None] * tmp2_NnFT[:, :, None]).sum(axis=1) # N F K T M
        self.W_noise_NnFK = self.W_noise_NnFK * self.xp.sqrt(a_W / b_W)
        self.H_noise_NnKT = self.H_noise_NnKT * self.xp.sqrt(a_H / b_H)

        self.lambda_NFT[1:] = self.W_noise_NnFK @ self.H_noise_NnKT
        self.Y_FTM = (self.lambda_NFT[..., None] * self.covarianceDiag_NFM[:, :, None]).sum(axis=0)


    def update_UV(self):
        a_1 = ((self.v_T[None] * self.power_speech_FT)[:, :, None] * self.Qx_power_FTM * self.covarianceDiag_NFM[0, :, None] / (self.Y_FTM ** 2)).sum(axis=2).sum(axis=1).real
        b_1 = ((self.v_T[None] * self.power_speech_FT)[:, :, None] * self.covarianceDiag_NFM[0, :, None] / self.Y_FTM).sum(axis=2).sum(axis=1).real
        self.u_F = self.u_F * self.xp.sqrt(a_1 / b_1)
        self.lambda_NFT[0] = self.u_F[:, None] * self.v_T[None] * self.power_speech_FT
        self.Y_FTM = (self.lambda_NFT[..., None] * self.covarianceDiag_NFM[:, :, None]).sum(axis=0)

        a_1 = ((self.u_F[:, None] * self.power_speech_FT)[:, :, None] * self.Qx_power_FTM * self.covarianceDiag_NFM[0, :, None] / (self.Y_FTM ** 2)).sum(axis=2).sum(axis=0).real
        b_1 = ((self.u_F[:, None] * self.power_speech_FT)[:, :, None] * self.covarianceDiag_NFM[0, :, None] / self.Y_FTM).sum(axis=2).sum(axis=0).real
        self.v_T = self.v_T * self.xp.sqrt(a_1 / b_1)
        self.lambda_NFT[0] = self.u_F[:, None] * self.v_T[None] * self.power_speech_FT
        self.Y_FTM = (self.lambda_NFT[..., None] * self.covarianceDiag_NFM[:, :, None]).sum(axis=0)


    def loss_func_Z(self, z, vae, n): # for update Z by backprop
        power_tmp_FT = chf.exp(vae.decode(z).T) + EPS
        Y_tmp_FTM = power_tmp_FT[:, :, None] * self.UVG_FTM+ self.WHG_noise_FTM
        return chf.sum(chf.log(Y_tmp_FTM) + self.Qx_power_FTM / Y_tmp_FTM ) / (self.NUM_freq * self.NUM_mic)


    def update_Z_speech(self, var_propose_distribution=1e-4):
        """
        Parameters:
            var_propose_distribution: float
                the variance of the propose distribution

        Results:
            self.Z_speech_DT: self.xp.array [ DIM_latent x T ]
                the latent variable of each speech
        """
        self.WHG_noise_FTM = (self.lambda_NFT[1:][..., None] * self.covarianceDiag_NFM[1:, :, None]).sum(axis=0)
        self.UVG_FTM = (self.u_F[:, None] * self.v_T[None])[:, :, None] * self.covarianceDiag_NFM[0, :, None]

        if "backprop" in self.MODE_update_Z: # acceptance rate is calculated from likelihood
            for it in range(self.NUM_Z_iteration):
                with chainer.using_config('train', False):
                    self.z_optimizer_speech.update(self.loss_func_Z, self.z_link_speech.z, self.speech_VAE, 0)

            self.Z_speech_DT = self.z_link_speech.z.data.T
            self.power_speech_FT = self.speech_VAE.decode_cupy(self.Z_speech_DT)

        if "sampling" in self.MODE_update_Z:
            log_var = self.xp.log(self.xp.ones_like(self.Z_speech_DT).astype(self.xp.float32) * var_propose_distribution)
            Z_speech_old_DT = self.Z_speech_DT
            power_old_FTM = self.speech_VAE.decode_cupy(Z_speech_old_DT)[:, :, None]

            for it in range(self.NUM_Z_iteration):
                Z_speech_new_DT = chf.gaussian(Z_speech_old_DT, log_var).data
                lambda_old_FTM = power_old_FTM * self.UVG_FTM + self.WHG_noise_FTM
                power_new_FTM = self.speech_VAE.decode_cupy(Z_speech_new_DT)[:, :, None]
                lambda_new_FTM = power_new_FTM * self.UVG_FTM + self.WHG_noise_FTM
                acceptance_rate = self.xp.exp((self.Qx_power_FTM * (1 / lambda_old_FTM - 1 / lambda_new_FTM)).sum(axis=2).sum(axis=0) + self.xp.log( ( lambda_old_FTM / lambda_new_FTM ).prod(axis=2).prod(axis=0) ) )
                accept_flag = self.xp.random.random([self.NUM_time]) < acceptance_rate
                Z_speech_old_DT[:, accept_flag] = Z_speech_new_DT[:, accept_flag]
                power_old_FTM[:, accept_flag] = power_new_FTM[:, accept_flag]

            self.Z_speech_DT = Z_speech_old_DT
            self.z_link_speech.z = chainer.Parameter(self.Z_speech_DT.T)
            self.power_speech_FT = self.speech_VAE.decode_cupy(self.Z_speech_DT)

        self.lambda_NFT[0] = self.u_F[:, None] * self.v_T[None] * self.power_speech_FT
        self.Y_FTM = (self.lambda_NFT[..., None] * self.covarianceDiag_NFM[:, :, None]).sum(axis=0)


    def save_parameter(self, fileName):
        param_list = [self.lambda_NFT, self.covarianceDiag_NFM, self.diagonalizer_FMM, self.u_F, self.v_T, self.Z_speech_DT, self.W_noise_NnFK, self.H_noise_NnKT]
        if self.xp != np:
            param_list = [self.convert_to_NumpyArray(param) for param in param_list]
        pic.dump(param_list, open(fileName, "wb"))


    def load_parameter(self, fileName):
        param_list = pic.load(open(fileName, "rb"))
        if self.xp != np:
            param_list = [cuda.to_gpu(param) for param in param_list]
        self.lambda_NFT, self.covarianceDiag_NFM, self.diagonalizer_FMM, self.u_F, self.v_T, self.Z_speech_DT, self.W_noise_NnFK, self.H_noise_NnKT = param_list


class Z_link(chainer.link.Link):
    def __init__(self, z):
        super(Z_link, self).__init__()

        with self.init_scope():
            self.z = chainer.Parameter(z)


if __name__ == "__main__":
    import soundfile as sf
    import librosa
    import sys, os
    from chainer import serializers
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(    'input_fileName', type= str, help='filename of the multichannel observed signals')
    parser.add_argument(         '--file_id', type= str, default="None", help='file id')
    parser.add_argument(             '--gpu', type= int, default=     0, help='GPU ID')
    parser.add_argument(      '--DIM_latent', type= int, default=   16, help='dimention of encoded vector')
    parser.add_argument(       '--NUM_noise', type= int, default=    1, help='number of noise')
    parser.add_argument(   '--NUM_iteration', type= int, default=  100, help='number of iteration')
    parser.add_argument( '--NUM_Z_iteration', type= int, default=   30, help='number of update Z iteration')
    parser.add_argument( '--NUM_basis_noise', type= int, default=   64, help='number of basis of noise (MODE_noise=NMF)')
    parser.add_argument(   '--MODE_update_Z', type= str, default="sampling", help='sampling, sampling2, backprop, backprop2, hybrid, hybrid2')
    parser.add_argument( '--MODE_initialize_covarianceMatrix', type=  str, default="obs", help='unit, obs')
    args = parser.parse_args()

    if args.gpu < 0:
        import numpy as xp
    else:
        import cupy as xp
        print("Use GPU " + str(args.gpu))
        chainer.cuda.get_device_from_id(args.gpu).use()

    sys.path.append("../DeepSpeechPrior")
    import network_VAE
    model_fileName = "../DeepSpeechPrior/model-VAE-best-scale=gamma-D={}.npz".format(args.DIM_latent)
    speech_VAE = network_VAE.VAE(n_latent=args.DIM_latent)
    serializers.load_npz(model_fileName, speech_VAE)
    name_DNN = "VAE"

    if xp != np:
        speech_VAE.to_gpu()

    wav, fs = sf.read(args.input_fileName)
    wav = wav.T
    M = len(wav)
    for m in range(M):
        tmp = librosa.core.stft(wav[m], n_fft=1024, hop_length=256)
        if m == 0:
            spec = np.zeros([tmp.shape[0], tmp.shape[1], M], dtype=np.complex)
        spec[:, :, m] = tmp

    separater = FastMNMF_DP(NUM_noise=args.NUM_noise, speech_VAE=speech_VAE, NUM_Z_iteration=args.NUM_Z_iteration, NUM_basis_noise=args.NUM_basis_noise, xp=xp, MODE_initialize_covarianceMatrix=args.MODE_initialize_covarianceMatrix, MODE_update_Z=args.MODE_update_Z)
    separater.load_spectrogram(spec)
    separater.file_id = args.file_id
    separater.name_DNN = name_DNN
    separater.solve(NUM_iteration=args.NUM_iteration, save_likelihood=False, save_parameter=False, save_path="./", interval_save_parameter=25)
