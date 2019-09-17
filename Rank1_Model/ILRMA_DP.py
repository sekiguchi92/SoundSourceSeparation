#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import chainer
from chainer import functions as chf
from chainer import cuda, serializers
from progressbar import progressbar
import librosa
import soundfile as sf
import sys, os
import pickle as pic

from configure import *
from ILRMA import ILRMA


class ILRMA_DP(ILRMA):
    def __init__(self, speech_VAE=None, NUM_basis_noise=2, xp=np, MODE_initialize_covarianceMatrix="unit", NUM_Z_iteration=30, MODE_update_Z="sampling", DIM_latent=16, normalize_encoder_input=True):
        """ initialize RANK1_MSE_DSP

        Parameters:
        -----------
            X: self.xp.array [ F * T * M ]
                power spectrogram of observed signals
            speech_VAE: VAE
                trained speech VAE network
            NUM_basis_noise: int
                the number of bases of each noise source
            MODE_initialize_covarianceMatrix: str
                how to initialize covariance matrix {unit, obs, cGMM}
            MODE_update_Z: str
                how to update latent variable Z {sampling, backprop}
            NUM_Z_iteration: int
                the number of iteration for updating Z per global iteration
            DIM_latent: int
                the dimension of latent variable Z
            normalize_encoder_input: boolean
                normalize observation to initialize latent variable by feeding the observation into a encoder
        """
        super(ILRMA_DP, self).__init__(xp=xp, MODE_initialize_covarianceMatrix=MODE_initialize_covarianceMatrix)
        self.method_name = "ILRMA_DP"
        self.NUM_basis_noise = NUM_basis_noise
        self.MODE_update_Z = MODE_update_Z
        self.NUM_Z_iteration = NUM_Z_iteration
        self.speech_VAE = speech_VAE
        self.xp = self.speech_VAE.xp
        self.normalize_encoder_input = normalize_encoder_input
        self.DIM_latent = DIM_latent
    

    def set_parameter(self, NUM_basis_noise=None, MODE_initialize_covarianceMatrix=None, NUM_Z_iteration=None, MODE_update_Z=None):
        if NUM_basis_noise != None:
            self.NUM_basis_noise = NUM_basis_noise
        if MODE_initialize_covarianceMatrix != None:
            self.MODE_initialize_covarianceMatrix = MODE_initialize_covarianceMatrix
        if NUM_Z_iteration != None:
            self.NUM_Z_iteration = NUM_Z_iteration
        if MODE_update_Z != None:
            self.MODE_update_Z = MODE_update_Z


    def initialize_PSD(self):
        self.NUM_speech, self.NUM_noise = 1, self.NUM_mic-1
        power_observation_FT = (self.xp.abs(self.X_FTM).astype(self.xp.float) ** 2).mean(axis=2)
        self.u_F = self.xp.ones(self.NUM_freq) / self.NUM_freq
        self.v_T = self.xp.ones(self.NUM_time)
        shape = 2
        self.W_noise_NnFK = self.xp.random.dirichlet(np.ones(self.NUM_freq)*shape, size=[self.NUM_noise, self.NUM_basis_noise]).transpose(0, 2, 1)
        self.H_noise_NnKT = self.xp.random.gamma(shape, (power_observation_FT.mean() * self.NUM_freq * self.NUM_mic / (self.NUM_noise * self.NUM_basis_noise)) / shape, size=[self.NUM_noise, self.NUM_basis_noise, self.NUM_time])
        self.H_noise_NnKT[self.H_noise_NnKT < EPS] = EPS

        if self.normalize_encoder_input:
            power_observation_FT = power_observation_FT / power_observation_FT.sum(axis=0).mean()
        self.Z_speech_DT = self.speech_VAE.encode_cupy(power_observation_FT.astype(self.xp.float32))
        self.z_link_speech = Z_link(self.Z_speech_DT.T)
        self.z_optimizer_speech = chainer.optimizers.Adam().setup(self.z_link_speech)
        self.power_speech_FT = self.speech_VAE.decode_cupy(self.Z_speech_DT)

        self.lambda_NFT = self.xp.zeros([self.NUM_source, self.NUM_freq, self.NUM_time]).astype(self.xp.float)
        self.lambda_NFT[0] = self.u_F[:, None] * self.v_T[None] * self.power_speech_FT
        self.lambda_NFT[1:] = self.W_noise_NnFK @ self.H_noise_NnKT


    def update(self):
        self.update_UV()
        self.update_WH_noise()
        self.update_Z_speech()
        self.update_SeparationMatrix()
        self.normalize()


    def make_filename_suffix(self):
        self.filename_suffix = "it={}-itZ={}-Ln={}-D={}-init={}-latent={}".format(self.NUM_iteration, self.NUM_Z_iteration, self.NUM_basis_noise, self.DIM_latent, self.MODE_initialize_covarianceMatrix, self.MODE_update_Z)
        if hasattr(self, "name_DNN"):
            self.filename_suffix += "-DNN={}".format(self.name_DNN)
        else:
            self.filename_suffix += "-DNN=NoName"
        if hasattr(self, "file_id"):
           self.filename_suffix += "-ID={}".format(self.file_id)
        else:
            print("====================\n\nWarning: Please set self.file_id\n\n====================")


    def normalize(self):
        mu_NF = self.xp.zeros([self.NUM_mic, self.NUM_freq])
        for m in range(self.NUM_mic):
            mu_NF[m] = (self.SeparationMatrix_FMM[:, m] * self.SeparationMatrix_FMM[:, m].conj()).sum(axis=1).real
            self.SeparationMatrix_FMM[:, m] = self.SeparationMatrix_FMM[:, m] / self.xp.sqrt(mu_NF[m, :, None])
        self.u_F = self.u_F / mu_NF[0]
        self.W_noise_NnFK = self.W_noise_NnFK / mu_NF[1:, :, None]

        nu = self.u_F.sum()
        self.u_F = self.u_F / nu
        self.v_T = nu * self.v_T

        nu_NnK = self.W_noise_NnFK.sum(axis=1)
        self.W_noise_NnFK = self.W_noise_NnFK / nu_NnK[:, None]
        self.H_noise_NnKT = self.H_noise_NnKT * nu_NnK[:, :, None]

        self.lambda_NFT[0] = self.u_F[:, None] * self.v_T[None] * self.power_speech_FT
        self.lambda_NFT[1:] = self.W_noise_NnFK @ self.H_noise_NnKT

        self.reset_variable()


    def update_UV(self):
        self.u_F = (self.Y_power_FTN[:, :, 0] / (self.v_T[None] * self.power_speech_FT)).mean(axis=1)
        self.v_T = (self.Y_power_FTN[:, :, 0] / (self.u_F[:, None] * self.power_speech_FT)).mean(axis=0)
        self.lambda_NFT[0] = self.u_F[:, None] * self.v_T[None] * self.power_speech_FT


    def update_WH_noise(self):
        """
        Results:
            self.W_noise_NnFK: self.xp.array [ NUM_noise x F x NUM_basis_noise ]
                the template of each basis
            self.H_noise_NnKT: self.xp.array [ NUM_noise x NUM_basis_noise x T ]
                the activation of each basis
        """
        if self.xp == np:
            for f in range(self.NUM_freq):
                numerator = (self.H_noise_NnKT * (self.Y_power_FTN[f, :, self.NUM_speech:].T / ( self.lambda_NFT[self.NUM_speech:, f] ** 2 ) )[:, None] ).sum(axis=2)
                denominator = ( self.H_noise_NnKT / self.lambda_NFT[self.NUM_speech:, f, None] ).sum(axis=2) # NUM_noise * NUM_basis_noise * F
                self.W_noise_NnFK[:, f] = self.W_noise_NnFK[:, f] * self.xp.sqrt(numerator / denominator)
            self.W_noise_NnFK[self.W_noise_NnFK < EPS] = EPS
            self.lambda_NFT[1:] = self.xp.matmul(self.W_noise_NnFK, self.H_noise_NnKT)

            numerator = self.xp.zeros_like(self.H_noise_NnKT)
            denominator = self.xp.zeros_like(self.H_noise_NnKT)
            for f in range(self.NUM_freq):
                numerator += self.W_noise_NnFK[:, f, :, None] * (self.Y_power_FTN[f, :, self.NUM_speech:].T / ( self.lambda_NFT[self.NUM_speech:, f] ** 2 ) )[:, None]
                denominator += self.W_noise_NnFK[:, f, :, None] / self.lambda_NFT[self.NUM_speech:, f, None] # NUM_noise * NUM_basis_noise * T
            self.H_noise_NnKT = self.H_noise_NnKT * self.xp.sqrt(numerator / denominator)
            self.H_noise_NnKT[self.H_noise_NnKT < EPS] = EPS
            self.lambda_NFT[1:] = self.xp.matmul(self.W_noise_NnFK, self.H_noise_NnKT)
        else:
            numerator = (self.H_noise_NnKT[:, None] * (self.Y_power_FTN.transpose(2, 0, 1)[self.NUM_speech:] / ( self.lambda_NFT[self.NUM_speech:] ** 2 ) )[:, :, None] ).sum(axis=3)
            denominator = ( self.H_noise_NnKT[:, None] / self.lambda_NFT[self.NUM_speech:][:, :, None] ).sum(axis=3) # NUM_noise * NUM_basis_noise * F
            self.W_noise_NnFK = self.W_noise_NnFK * self.xp.sqrt(numerator / denominator)
            self.W_noise_NnFK[self.W_noise_NnFK < EPS] = EPS
            self.lambda_NFT[1:] = self.xp.matmul(self.W_noise_NnFK, self.H_noise_NnKT)

            numerator = (self.W_noise_NnFK[:, :, :, None] * (self.Y_power_FTN.transpose(2, 0, 1)[self.NUM_speech] / ( self.lambda_NFT[self.NUM_speech:] ** 2 ) )[:, :, None] ).sum(axis=1)
            denominator = ( self.W_noise_NnFK[:, :, :, None] / self.lambda_NFT[self.NUM_speech:][:, :, None] ).sum(axis=1) # NUM_noise * NUM_basis_noise * T
            self.H_noise_NnKT = self.H_noise_NnKT * self.xp.sqrt(numerator / denominator)
            self.H_noise_NnKT[self.H_noise_NnKT < EPS] = EPS
            self.lambda_NFT[1:] = self.xp.matmul(self.W_noise_NnFK, self.H_noise_NnKT)


    def calculate_log_likelihood(self):
        return -1 * (self.Y_power_FTN.transpose(2, 0, 1) / self.lambda_NFT + self.xp.log(self.lambda_NFT)).sum() + self.NUM_time * np.log(np.linalg.det(self.convert_to_NumpyArray(self.SeparationMatrix_FMM @ self.SeparationMatrix_FMM.conj().transpose(0, 2, 1)))).sum().real


    def loss_func_Z(self, z, vae, n):
        power_FT = chf.exp(vae.decode(z).T) * self.u_F[:, None] * self.v_T[None] + EPS
        loss = chf.sum(self.Y_power_FTN[:, :, n] / power_FT + chf.log(power_FT))
        return loss


    def update_Z_speech(self, var_propose_distribution=1e-3):
        """
        Parameters:
            iteration: int
                the number of iteration to update Z
            var_propose_distribution: float
                the variance of the propose distribution

        Results:
            self.Z_speech_DT: self.xp.array [ DIM_latent x T ]
                the latent variable of each speech
        """
        if (self.MODE_update_Z == "backprop"): # acceptance rate is calculated from likelihood
            for it in range(self.NUM_Z_iteration):
                with chainer.using_config('train', False):
                    self.z_optimizer_speech.update(self.loss_func_Z, self.z_link_speech.z, self.speech_VAE, 0)

            self.Z_speech_DT = self.z_link_speech.z.data.T
            self.power_speech_FT = self.speech_VAE.decode_cupy(self.Z_speech_DT)


        if ("sampling" in self.MODE_update_Z):
            log_var = self.xp.log(self.xp.ones_like(self.Z_speech_DT).astype(self.xp.float32) * var_propose_distribution)
            Z_old_DT = self.Z_speech_DT
            power_old_FT = self.speech_VAE.decode_cupy(Z_old_DT)
            for it in range(self.NUM_Z_iteration):
                Z_new_DT = chf.gaussian(Z_old_DT, log_var).data
                power_new_FT = self.speech_VAE.decode_cupy(Z_new_DT)
                acceptance_rate = self.xp.exp(( self.Y_power_FTN[:, :, 0] * (1 / (power_old_FT * self.u_F[:, None] * self.v_T[None]) - 1 / (power_new_FT * self.u_F[:, None] * self.v_T[None]) ) + self.xp.log( power_old_FT / power_new_FT) ).sum(axis=0) - (Z_new_DT ** 2 - Z_old_DT ** 2).sum(axis=0)/2)
                acceptance_boolean = self.xp.random.random([self.NUM_time]) < acceptance_rate
                Z_old_DT[:, acceptance_boolean] = Z_new_DT[:, acceptance_boolean]
                power_old_FT[:, acceptance_boolean] = power_new_FT[:, acceptance_boolean]

            self.Z_speech_DT = Z_old_DT
            self.z_link_speech.z = chainer.Parameter(self.Z_speech_DT.T)
            self.power_speech_FT = self.speech_VAE.decode_cupy(self.Z_speech_DT)

        self.lambda_NFT[0] = self.u_F[:, None] * self.v_T[None] * self.power_speech_FT


    def save_parameter(self, filename):
        param_list = [self.SeparationMatrix_FMM, self.power_speech_FT, self.u_F, self.v_T, self.Z_speech_DT, self.W_noise_NnFK, self.H_noise_NnKT]

        if self.xp != np:
            param_list = [self.convert_to_NumpyArray(param) for param in param_list]

        pic.dump(param_list, open(filename, "wb"))


    def load_parameter(self, filename):
        param_list = pic.load(open(filename, "rb"))
        if self.xp != np:
            param_list = [cuda.to_gpu(param) for param in param_list]
        self.SeparationMatrix_FMM, self.power_speech_FT, self.u_F, self.v_T, self.Z_speech_DT, self.W_noise_NnFK, self.H_noise_NnKT = param_list



class Z_link(chainer.link.Link):
    def __init__(self, z):
        super(Z_link, self).__init__()

        with self.init_scope():
            self.z = chainer.Parameter(z)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(    'input_fileName', type= str, help='filename of the multichannel observed signals')
    parser.add_argument(         '--file_id', type= str, default="None", help='file id')
    parser.add_argument(             '--gpu', type= int, default=     0, help='GPU ID')
    parser.add_argument(      '--DIM_latent', type= int, default=   16, help='dimention of encoded vector')
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

    separater = ILRMA_DP(speech_VAE=speech_VAE, NUM_Z_iteration=args.NUM_Z_iteration, NUM_basis_noise=args.NUM_basis_noise, xp=xp, MODE_initialize_covarianceMatrix=args.MODE_initialize_covarianceMatrix, MODE_update_Z=args.MODE_update_Z)
    separater.load_spectrogram(spec)
    separater.file_id = args.file_id
    separater.name_DNN = name_DNN
    separater.solve(NUM_iteration=args.NUM_iteration, save_likelihood=False, save_parameter=False, save_path="./", interval_save_parameter=25)
