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
    """ Blind Speech Enhancement Using Independent Low-rank Matrix Analysis with a Deep Speech Prior (ILRMA-DP)

    X_FTM: the observed complex spectrogram
    W_noise_NnFK: basis vectors for noise sources (Nn means the number of noise sources)
    H_noise_NnKT: activations for noise sources
    Z_speech_DT: latent variables for speech
    power_speech_FT: power spectra of speech that is the output of DNN(Z_speech_DT)
    lambda_NFT: power spectral densities of each source
        lambda_NFT[0] = U_F * V_T * power_speech_FT
        lambda_NFT[1:] = W_noise_NnFK @ H_noise_NnKT
    SeparationMatrix_FMM: separation matrices
    """

    def __init__(self, speech_VAE=None, n_basis_noise=2, xp=np, init_SCM="unit", n_Z_iteration=30, mode_update_Z="sampling",\
            n_latent=16, normalize_encoder_input=True, n_bit=64, seed=0):
        """ initialize RANK1_MSE_DSP

        Parameters:
        -----------
            X: self.xp.array [ F * T * M ]
                power spectrogram of observed signals
            speech_VAE: VAE
                trained speech VAE network
            n_basis_noise: int
                the number of bases of each noise source
            init_SCM: str
                how to initialize covariance matrix {unit, obs, cGMM}
            mode_update_Z: str
                how to update latent variable Z {sampling, backprop}
            n_Z_iteration: int
                the number of iteration for updating Z per global iteration
            n_latent: int
                the dimension of latent variable Z
            normalize_encoder_input: boolean
                normalize observation to initialize latent variable by feeding the observation into a encoder
            n_bit:int (32 or 64)
                The number of bits for floating point number.
                '32' may degrade the peformance in exchange for lower computational cost.
                32 -> float32 and complex64
                64 -> float64 and complex128
        """
        super(ILRMA_DP, self).__init__(xp=xp, init_SCM=init_SCM, n_bit=n_bit, seed=seed)
        self.method_name = "ILRMA_DP"
        self.n_basis_noise = n_basis_noise
        self.mode_update_Z = mode_update_Z
        self.n_Z_iteration = n_Z_iteration
        self.speech_VAE = speech_VAE
        self.xp = self.speech_VAE.xp
        self.normalize_encoder_input = normalize_encoder_input
        self.n_latent = n_latent


    def initialize_PSD(self):
        self.n_speech, self.n_noise = 1, self.n_mic-1
        self.W_noise_NnFK = self.xp.random.rand(self.n_noise, self.n_freq, self.n_basis_noise).astype(self.TYPE_FLOAT)
        self.H_noise_NnKT = self.xp.random.rand(self.n_noise, self.n_basis_noise, self.n_time).astype(self.TYPE_FLOAT)
        self.U_F = self.xp.ones(self.n_freq, dtype=self.TYPE_FLOAT) / self.n_freq
        self.V_T = self.xp.ones(self.n_time, dtype=self.TYPE_FLOAT)

        power_observation_FT = (self.xp.abs(self.X_FTM).astype(self.xp.float) ** 2).mean(axis=2)
        if self.normalize_encoder_input:
            power_observation_FT = power_observation_FT / power_observation_FT.sum(axis=0).mean()
        self.Z_speech_DT = self.speech_VAE.encode_cupy(power_observation_FT.astype(self.xp.float32))
        self.z_link_speech = Z_link(self.Z_speech_DT.T)
        self.z_optimizer_speech = chainer.optimizers.Adam().setup(self.z_link_speech)
        self.power_speech_FT = self.speech_VAE.decode_cupy(self.Z_speech_DT)

        self.lambda_NFT = self.xp.zeros([self.n_source, self.n_freq, self.n_time]).astype(self.TYPE_FLOAT)
        self.lambda_NFT[0] = self.U_F[:, None] * self.V_T[None] * self.power_speech_FT
        self.lambda_NFT[1:] = self.W_noise_NnFK @ self.H_noise_NnKT


    def update(self):
        self.update_UV()
        self.update_WH_noise()
        self.update_Z_speech()
        self.update_SeparationMatrix()
        self.normalize()


    def make_filename_suffix(self):
        self.filename_suffix = f"it={self.n_iteration}-itZ={self.n_Z_iteration}-Ln={self.n_basis_noise}-D={self.n_latent}-init={self.init_SCM}-latent={self.mode_update_Z}"

        if hasattr(self, "name_DNN"):
            self.filename_suffix += f"-DNN={self.name_DNN}"
        if self.n_bit != 64:
            self.filename_suffix += f"-bit={self.n_bit}"
        if hasattr(self, "file_id"):
            self.filename_suffix += f"-ID={self.file_id}"
        print("param:", self.filename_suffix)


    def normalize(self):
        mu_NF = self.xp.zeros([self.n_mic, self.n_freq], dtype=self.TYPE_FLOAT)
        for m in range(self.n_mic):
            mu_NF[m] = (self.SeparationMatrix_FMM[:, m] * self.SeparationMatrix_FMM[:, m].conj()).sum(axis=1).real
            self.SeparationMatrix_FMM[:, m] /= self.xp.sqrt(mu_NF[m, :, None])
        self.U_F /= mu_NF[0]
        self.W_noise_NnFK /= mu_NF[1:, :, None]

        nu = self.U_F.sum()
        self.U_F /= nu
        self.V_T *= nu

        nu_NnK = self.W_noise_NnFK.sum(axis=1)
        self.W_noise_NnFK /= nu_NnK[:, None]
        self.H_noise_NnKT *= nu_NnK[:, :, None]

        self.lambda_NFT[0] = self.U_F[:, None] * self.V_T[None] * self.power_speech_FT
        self.lambda_NFT[1:] = self.W_noise_NnFK @ self.H_noise_NnKT

        self.reset_variable()


    def update_UV(self):
        self.U_F = (self.Y_power_FTN[:, :, 0] / (self.V_T[None] * self.power_speech_FT)).mean(axis=1)
        self.V_T = (self.Y_power_FTN[:, :, 0] / (self.U_F[:, None] * self.power_speech_FT)).mean(axis=0)
        self.lambda_NFT[0] = self.U_F[:, None] * self.V_T[None] * self.power_speech_FT


    def update_WH_noise(self):
        """
        Results:
            self.W_noise_NnFK: self.xp.array [ n_noise x F x n_basis_noise ]
                the template of each basis
            self.H_noise_NnKT: self.xp.array [ n_noise x n_basis_noise x T ]
                the activation of each basis
        """
        numerator = (self.H_noise_NnKT[:, None] * (self.Y_power_FTN.transpose(2, 0, 1)[self.n_speech:] / ( self.lambda_NFT[self.n_speech:] ** 2 ) )[:, :, None] ).sum(axis=3)
        denominator = ( self.H_noise_NnKT[:, None] / self.lambda_NFT[self.n_speech:][:, :, None] ).sum(axis=3) # n_noise * n_basis_noise * F
        self.W_noise_NnFK = self.W_noise_NnFK * self.xp.sqrt(numerator / denominator)
        self.W_noise_NnFK[self.W_noise_NnFK < EPS] = EPS
        self.lambda_NFT[1:] = self.xp.matmul(self.W_noise_NnFK, self.H_noise_NnKT)

        numerator = (self.W_noise_NnFK[:, :, :, None] * (self.Y_power_FTN.transpose(2, 0, 1)[self.n_speech] / ( self.lambda_NFT[self.n_speech:] ** 2 ) )[:, :, None] ).sum(axis=1)
        denominator = ( self.W_noise_NnFK[:, :, :, None] / self.lambda_NFT[self.n_speech:][:, :, None] ).sum(axis=1) # n_noise * n_basis_noise * T
        self.H_noise_NnKT = self.H_noise_NnKT * self.xp.sqrt(numerator / denominator)
        self.H_noise_NnKT[self.H_noise_NnKT < EPS] = EPS
        self.lambda_NFT[1:] = self.xp.matmul(self.W_noise_NnFK, self.H_noise_NnKT)


    def calculate_log_likelihood(self):
        return -1 * (self.Y_power_FTN.transpose(2, 0, 1) / self.lambda_NFT + self.xp.log(self.lambda_NFT)).sum() + self.n_time * np.log(np.linalg.det(self.convert_to_NumpyArray(self.SeparationMatrix_FMM @ self.SeparationMatrix_FMM.conj().transpose(0, 2, 1)))).sum().real


    def loss_func_Z(self, z, vae, n):
        power_FT = chf.exp(vae.decode(z).T) * self.U_F[:, None] * self.V_T[None] + EPS
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
            self.Z_speech_DT: self.xp.array [ n_latent x T ]
                the latent variable of each speech
        """
        if (self.mode_update_Z == "backprop"): # acceptance rate is calculated from likelihood
            for it in range(self.n_Z_iteration):
                with chainer.using_config('train', False):
                    self.z_optimizer_speech.update(self.loss_func_Z, self.z_link_speech.z, self.speech_VAE, 0)

            self.Z_speech_DT = self.z_link_speech.z.data.T
            self.power_speech_FT = self.speech_VAE.decode_cupy(self.Z_speech_DT)


        if ("sampling" in self.mode_update_Z):
            log_var = self.xp.log(self.xp.ones_like(self.Z_speech_DT).astype(self.xp.float32) * var_propose_distribution)
            Z_old_DT = self.Z_speech_DT
            power_old_FT = self.speech_VAE.decode_cupy(Z_old_DT)
            for it in range(self.n_Z_iteration):
                Z_new_DT = chf.gaussian(Z_old_DT, log_var).data
                power_new_FT = self.speech_VAE.decode_cupy(Z_new_DT)
                acceptance_rate = self.xp.exp(( self.Y_power_FTN[:, :, 0] * (1 / (power_old_FT * self.U_F[:, None] * self.V_T[None]) - 1 / (power_new_FT * self.U_F[:, None] * self.V_T[None]) ) + self.xp.log( power_old_FT / power_new_FT) ).sum(axis=0) - (Z_new_DT ** 2 - Z_old_DT ** 2).sum(axis=0)/2)
                acceptance_boolean = self.xp.random.random([self.n_time]) < acceptance_rate
                Z_old_DT[:, acceptance_boolean] = Z_new_DT[:, acceptance_boolean]
                power_old_FT[:, acceptance_boolean] = power_new_FT[:, acceptance_boolean]

            self.Z_speech_DT = Z_old_DT
            self.z_link_speech.z = chainer.Parameter(self.Z_speech_DT.T)
            self.power_speech_FT = self.speech_VAE.decode_cupy(self.Z_speech_DT)

        self.lambda_NFT[0] = self.U_F[:, None] * self.V_T[None] * self.power_speech_FT


    def save_parameter(self, filename):
        param_list = [self.SeparationMatrix_FMM, self.power_speech_FT, self.U_F, self.V_T, self.Z_speech_DT, self.W_noise_NnFK, self.H_noise_NnKT]

        if self.xp != np:
            param_list = [self.convert_to_NumpyArray(param) for param in param_list]

        pic.dump(param_list, open(filename, "wb"))


    def load_parameter(self, filename):
        param_list = pic.load(open(filename, "rb"))
        if self.xp != np:
            param_list = [cuda.to_gpu(param) for param in param_list]
        self.SeparationMatrix_FMM, self.power_speech_FT, self.U_F, self.V_T, self.Z_speech_DT, self.W_noise_NnFK, self.H_noise_NnKT = param_list



class Z_link(chainer.link.Link):
    def __init__(self, z):
        super(Z_link, self).__init__()

        with self.init_scope():
            self.z = chainer.Parameter(z)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(  'input_fileName', type= str, help='filename of the multichannel observed signals')
    parser.add_argument(       '--file_id', type= str, default=    "None", help='file id')
    parser.add_argument(           '--gpu', type= int, default=         0, help='GPU ID')
    parser.add_argument(      '--n_latent', type= int, default=        16, help='dimention of encoded vector')
    parser.add_argument(      '--init_SCM', type= str, default=     "obs", help='unit, obs')
    parser.add_argument(   '--n_iteration', type= int, default=       100, help='number of iteration')
    parser.add_argument( '--n_Z_iteration', type= int, default=        30, help='number of update Z iteration')
    parser.add_argument( '--n_basis_noise', type= int, default=        64, help='number of basis of noise (MODE_noise=NMF)')
    parser.add_argument( '--mode_update_Z', type= str, default="sampling", help='sampling, sampling2, backprop, backprop2, hybrid, hybrid2')
    parser.add_argument(         '--n_bit', type= int, default=        64, help='number of bits for floating point number')
    args = parser.parse_args()

    if args.gpu < 0:
        import numpy as xp
    else:
        import cupy as xp
        print("Use GPU " + str(args.gpu))
        chainer.cuda.get_device_from_id(args.gpu).use()

    sys.path.append("../DeepSpeechPrior")
    import network_VAE
    model_fileName = "../DeepSpeechPrior/model-VAE-best-scale=gamma-D={}.npz".format(args.n_latent)
    speech_VAE = network_VAE.VAE(n_latent=args.n_latent)
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

    separater = ILRMA_DP(speech_VAE=speech_VAE, n_Z_iteration=args.n_Z_iteration, n_basis_noise=args.n_basis_noise, xp=xp, init_SCM=args.init_SCM, mode_update_Z=args.mode_update_Z, n_bit=args.n_bit, seed=0)
    separater.load_spectrogram(spec)
    separater.file_id = args.file_id
    separater.name_DNN = name_DNN
    separater.solve(n_iteration=args.n_iteration, save_likelihood=False, save_parameter=False, save_path="./", interval_save_parameter=25)
