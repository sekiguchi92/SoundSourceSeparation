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
    """ Blind Speech Enhancement Using Fast Multichannel NMF with a Deep Speech Prior (FastMNMF_DP)

    X_FTM: the observed complex spectrogram
    Q_FMM: diagonalizer that converts a spatial covariance matrix (SCM) to a diagonal matrix
    G_NFM: diagonal elements of the diagonalized SCMs (N means the number of all sources)
    W_noise_NnFK: basis vectors for noise sources (Nn means the number of noise sources)
    H_noise_NnKT: activations for noise sources
    Z_speech_DT: latent variables for speech
    power_speech_FT: power spectra of speech that is the output of DNN(Z_speech_DT)
    lambda_NFT: power spectral densities of each source
        lambda_NFT[0] = U_F * V_T * power_speech_FT
        lambda_NFT[1:] = W_noise_NnFK @ H_noise_NnKT
    Qx_power_FTM: power spectra of Qx
    Y_FTM: \sum_n lambda_NFT G_NFM
    """

    def __init__(self, speech_VAE=None, n_noise=1, n_Z_iteration=30, n_latent=16, n_basis_noise=2, xp=np, \
            init_SCM="circular", mode_update_Z="sampling", normalize_encoder_input=True, n_bit=64, seed=0):
        """ initialize FastMNMF_DP

        Parameters:
        -----------
            n_noise: int
                The number of noise sources
            speech_VAE: VAE
                Trained speech VAE network (necessary if you use VAE as speech model)
            n_latent: int
                The dimension of latent variable Z
            n_basis_noise: int
                The number of bases of each noise source
            init_SCM: str
                How to initialize covariance matrix {circular, gradual, obs, ILRMA}
                About circular and gradual initialization, please check my paper:
                    Kouhei Sekiguchi, Yoshiaki Bando, Aditya Arie Nugraha, Kazuyoshi Yoshii, Tatsuya Kawahara:
                    Fast Multichannel Nonnegative Matrix Factorization with Directivity-Aware
                        Jointly-Diagonalizable Spatial Covariance Matrices for Blind Source Separation,
                    IEEE/ACM Transactions on Audio, Speech, and Language Processing, accepted, 2020.
            mode_update_Z: str
                How to update latent variable Z {sampling, backprop}
            normalize_encoder_input: boolean
                Whether observation X is normalized or not
                When the latent variable Z is initialized with the output of VAE given X.
                Since the default VAE is trained with normalized data, default is True.
            n_bit:int (32 or 64)
                The number of bits for floating point number.
                '32' may degrade the peformance in exchange for lower computational cost.
                32 -> float32 and complex64
                64 -> float64 and complex128
        """
        super(FastMNMF_DP, self).__init__(n_source=n_noise+1, xp=xp, init_SCM=init_SCM, n_bit=n_bit, seed=seed)
        self.method_name = "FastMNMF_DP"
        self.n_source, self.n_speech, self.n_noise = n_noise+1, 1, n_noise
        self.speech_VAE = speech_VAE
        self.n_Z_iteration = n_Z_iteration
        self.n_basis_noise = n_basis_noise
        self.n_latent = n_latent
        self.mode_update_Z = mode_update_Z
        self.normalize_encoder_input = normalize_encoder_input


    def initialize_PSD(self):
        """
        initialize parameters related to power spectral density (PSD)
        W, H, U, V, Z
        """
        self.W_noise_NnFK = self.xp.random.rand(self.n_noise, self.n_freq, self.n_basis_noise).astype(self.TYPE_FLOAT)
        self.H_noise_NnKT = self.xp.random.rand(self.n_noise, self.n_basis_noise, self.n_time).astype(self.TYPE_FLOAT)

        self.U_F = self.xp.ones(self.n_freq, dtype=self.TYPE_FLOAT) / self.n_freq
        self.V_T = self.xp.ones(self.n_time, dtype=self.TYPE_FLOAT)

        power_observation_FT = (self.xp.abs(self.X_FTM) ** 2).mean(axis=2)
        if self.normalize_encoder_input:
            power_observation_FT = power_observation_FT / power_observation_FT.sum(axis=0).mean()

        self.Z_speech_DT = self.speech_VAE.encode_cupy(power_observation_FT.astype(self.xp.float32))
        self.z_link_speech = Z_link(self.Z_speech_DT.T)
        self.z_optimizer_speech = chainer.optimizers.Adam().setup(self.z_link_speech)
        self.power_speech_FT = self.speech_VAE.decode_cupy(self.Z_speech_DT)

        self.lambda_NFT = self.xp.zeros([self.n_source, self.n_freq, self.n_time], dtype=self.TYPE_FLOAT)
        self.lambda_NFT[0] = self.U_F[:, None] * self.V_T[None] * self.power_speech_FT
        self.lambda_NFT[1:] = self.W_noise_NnFK @ self.H_noise_NnKT


    def make_filename_suffix(self):
        self.filename_suffix = f"S={self.n_source}-it={self.n_iteration}-itZ={self.n_Z_iteration}" \
            + f"-Ln={self.n_basis_noise}-D={self.n_latent}-init={self.init_SCM}-latent={self.mode_update_Z}"

        if hasattr(self, "name_DNN"):
            self.filename_suffix += f"-DNN={self.name_DNN}"
        if self.n_bit != 64:
            self.filename_suffix += f"-bit={self.n_bit}"
        if hasattr(self, "file_id"):
            self.filename_suffix += f"-ID={self.file_id}"
        print("param:", self.filename_suffix)


    def update(self):
        self.update_UV()
        self.update_Z_speech()
        self.update_WH_noise()
        self.update_CovarianceDiagElement()
        self.udpate_Diagonalizer()
        self.normalize()


    def normalize(self):
        phi_F = self.xp.sum(self.Q_FMM * self.Q_FMM.conj(), axis=(1, 2)).real / self.n_mic
        self.Q_FMM /= self.xp.sqrt(phi_F)[:, None, None]
        self.G_NFM /= phi_F[None, :, None]

        mu_NF = (self.G_NFM).sum(axis=2).real
        self.G_NFM /= mu_NF[:, :, None]
        self.U_F *= mu_NF[0]
        self.W_noise_NnFK *= mu_NF[1:][:, :, None]

        nu = self.U_F.sum()
        self.U_F /= nu
        self.V_T *= nu
        self.lambda_NFT[0] = self.U_F[:, None] * self.V_T[None] * self.power_speech_FT

        nu_NnK = self.W_noise_NnFK.sum(axis=1)
        self.W_noise_NnFK /= nu_NnK[:, None]
        self.H_noise_NnKT *= nu_NnK[:, :, None]
        self.lambda_NFT[1:] = self.W_noise_NnFK @ self.H_noise_NnKT + EPS

        self.reset_variable()


    def update_WH_noise(self):
        tmp1_NnFT = (self.G_NFM[1, :, None] * (self.Qx_power_FTM / (self.Y_FTM ** 2))[None]).sum(axis=3)
        tmp2_NnFT = (self.G_NFM[1, :, None] / self.Y_FTM[None]).sum(axis=3)
        a_W = (self.H_noise_NnKT[:, None] * tmp1_NnFT[:, :, None]).sum(axis=3)  # N F K T M
        b_W = (self.H_noise_NnKT[:, None] * tmp2_NnFT[:, :, None]).sum(axis=3)
        a_H = (self.W_noise_NnFK[..., None] * tmp1_NnFT[:, :, None] ).sum(axis=1) # N F K T M
        b_H = (self.W_noise_NnFK[..., None] * tmp2_NnFT[:, :, None]).sum(axis=1) # N F K T M
        self.W_noise_NnFK *= self.xp.sqrt(a_W / b_W)
        self.H_noise_NnKT *= self.xp.sqrt(a_H / b_H)

        self.lambda_NFT[1:] = self.W_noise_NnFK @ self.H_noise_NnKT + EPS
        self.Y_FTM = (self.lambda_NFT[..., None] * self.G_NFM[:, :, None]).sum(axis=0)


    def update_UV(self):
        a_1 = ((self.V_T[None] * self.power_speech_FT)[:, :, None] * self.Qx_power_FTM * self.G_NFM[0, :, None] / (self.Y_FTM ** 2)).sum(axis=2).sum(axis=1).real
        b_1 = ((self.V_T[None] * self.power_speech_FT)[:, :, None] * self.G_NFM[0, :, None] / self.Y_FTM).sum(axis=2).sum(axis=1).real
        self.U_F *= self.xp.sqrt(a_1 / b_1)
        self.lambda_NFT[0] = self.U_F[:, None] * self.V_T[None] * self.power_speech_FT
        self.Y_FTM = (self.lambda_NFT[..., None] * self.G_NFM[:, :, None]).sum(axis=0)

        a_1 = ((self.U_F[:, None] * self.power_speech_FT)[:, :, None] * self.Qx_power_FTM * self.G_NFM[0, :, None] / (self.Y_FTM ** 2)).sum(axis=2).sum(axis=0).real
        b_1 = ((self.U_F[:, None] * self.power_speech_FT)[:, :, None] * self.G_NFM[0, :, None] / self.Y_FTM).sum(axis=2).sum(axis=0).real
        self.V_T *= self.xp.sqrt(a_1 / b_1)
        self.lambda_NFT[0] = self.U_F[:, None] * self.V_T[None] * self.power_speech_FT
        self.Y_FTM = (self.lambda_NFT[..., None] * self.G_NFM[:, :, None]).sum(axis=0)


    def loss_func_Z(self, z, vae, n): # for update Z by backprop
        power_tmp_FT = chf.exp(vae.decode(z).T) + EPS
        Y_tmp_FTM = power_tmp_FT[:, :, None] * self.UVG_FTM+ self.WHG_noise_FTM
        return chf.sum(chf.log(Y_tmp_FTM) + self.Qx_power_FTM / Y_tmp_FTM ) / (self.n_freq * self.n_mic)


    def update_Z_speech(self, var_propose_distribution=1e-4):
        """
        Parameters:
            var_propose_distribution: float
                the variance of the propose distribution

        Results:
            self.Z_speech_DT: self.xp.array [ n_latent x T ]
                the latent variable of each speech
        """
        self.WHG_noise_FTM = (self.lambda_NFT[1:][..., None] * self.G_NFM[1:, :, None]).sum(axis=0)
        self.UVG_FTM = (self.U_F[:, None] * self.V_T[None])[:, :, None] * self.G_NFM[0, :, None]

        if "backprop" in self.mode_update_Z: # acceptance rate is calculated from likelihood
            for it in range(self.n_Z_iteration):
                with chainer.using_config('train', False):
                    self.z_optimizer_speech.update(self.loss_func_Z, self.z_link_speech.z, self.speech_VAE, 0)

            self.Z_speech_DT = self.z_link_speech.z.data.T
            self.power_speech_FT = self.speech_VAE.decode_cupy(self.Z_speech_DT)

        if "sampling" in self.mode_update_Z:
            log_var = self.xp.log(self.xp.ones_like(self.Z_speech_DT).astype(self.xp.float32) * var_propose_distribution)
            Z_speech_old_DT = self.Z_speech_DT
            power_old_FTM = self.speech_VAE.decode_cupy(Z_speech_old_DT)[:, :, None]

            for it in range(self.n_Z_iteration):
                Z_speech_new_DT = chf.gaussian(Z_speech_old_DT, log_var).data
                lambda_old_FTM = power_old_FTM * self.UVG_FTM + self.WHG_noise_FTM
                power_new_FTM = self.speech_VAE.decode_cupy(Z_speech_new_DT)[:, :, None]
                lambda_new_FTM = power_new_FTM * self.UVG_FTM + self.WHG_noise_FTM
                acceptance_rate = self.xp.exp((self.Qx_power_FTM * (1 / lambda_old_FTM - 1 / lambda_new_FTM)).sum(axis=2).sum(axis=0) + self.xp.log( ( lambda_old_FTM / lambda_new_FTM ).prod(axis=2).prod(axis=0) ) )
                accept_flag = self.xp.random.random([self.n_time]) < acceptance_rate
                Z_speech_old_DT[:, accept_flag] = Z_speech_new_DT[:, accept_flag]
                power_old_FTM[:, accept_flag] = power_new_FTM[:, accept_flag]

            self.Z_speech_DT = Z_speech_old_DT
            self.z_link_speech.z = chainer.Parameter(self.Z_speech_DT.T)
            self.power_speech_FT = self.speech_VAE.decode_cupy(self.Z_speech_DT)

        self.lambda_NFT[0] = self.U_F[:, None] * self.V_T[None] * self.power_speech_FT
        self.Y_FTM = (self.lambda_NFT[..., None] * self.G_NFM[:, :, None]).sum(axis=0)


    def save_parameter(self, filename):
        param_list = [self.lambda_NFT, self.G_NFM, self.Q_FMM, self.U_F, self.V_T, self.Z_speech_DT, self.W_noise_NnFK, self.H_noise_NnKT]
        if self.xp != np:
            param_list = [self.convert_to_NumpyArray(param) for param in param_list]
        pic.dump(param_list, open(filename, "wb"))


    def load_parameter(self, filename):
        param_list = pic.load(open(filename, "rb"))
        if self.xp != np:
            param_list = [cuda.to_gpu(param) for param in param_list]
        self.lambda_NFT, self.G_NFM, self.Q_FMM, self.U_F, self.V_T, self.Z_speech_DT, self.W_noise_NnFK, self.H_noise_NnKT = param_list


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
    parser.add_argument( 'input_filename', type= str, help='filename of the multichannel observed signals')
    parser.add_argument(      '--file_id', type= str, default=    "None", help='file id')
    parser.add_argument(          '--gpu', type= int, default=         0, help='GPU ID')
    parser.add_argument(     '--n_latent', type= int, default=        16, help='dimention of encoded vector')
    parser.add_argument(      '--n_noise', type= int, default=         1, help='number of noise')
    parser.add_argument('--n_basis_noise', type= int, default=        64, help='number of basis of noise (MODE_noise=NMF)')
    parser.add_argument(     '--init_SCM', type= str, default=     "obs", help='obs (for speech enhancement)')
    parser.add_argument(  '--n_iteration', type= int, default=       100, help='number of iteration')
    parser.add_argument('--n_Z_iteration', type= int, default=        30, help='number of update Z iteration')
    parser.add_argument('--mode_update_Z', type= str, default="sampling", help='sampling, backprop')
    parser.add_argument(        '--n_mic', type= int, default=         8, help='number of microphones')
    parser.add_argument(        '--n_bit', type= int, default=        64, help='number of bits for floating point number')
    args = parser.parse_args()

    if args.gpu < 0:
        import numpy as xp
    else:
        import cupy as xp
        print("Use GPU " + str(args.gpu))
        chainer.cuda.get_device_from_id(args.gpu).use()

    sys.path.append("../DeepSpeechPrior")
    import network_VAE
    model_filename = f"../DeepSpeechPrior/model-VAE-best-scale=gamma-D={args.n_latent}.npz"
    speech_VAE = network_VAE.VAE(n_latent=args.n_latent)
    serializers.load_npz(model_filename, speech_VAE)
    name_DNN = "VAE"

    if xp != np:
        speech_VAE.to_gpu()

    wav, fs = sf.read(args.input_filename)
    wav = wav.T
    M = min(args.n_mic, len(wav))
    for m in range(M):
        tmp = librosa.core.stft(wav[m], n_fft=1024, hop_length=256)
        if m == 0:
            spec = np.zeros([tmp.shape[0], tmp.shape[1], M], dtype=np.complex)
        spec[:, :, m] = tmp

    separater = FastMNMF_DP(n_noise=args.n_noise, speech_VAE=speech_VAE, n_Z_iteration=args.n_Z_iteration, n_basis_noise=args.n_basis_noise, xp=xp, init_SCM=args.init_SCM, mode_update_Z=args.mode_update_Z, n_bit=args.n_bit, seed=0)
    separater.load_spectrogram(spec)
    separater.file_id = args.file_id
    separater.name_DNN = name_DNN
    separater.solve(n_iteration=args.n_iteration, save_likelihood=False, save_parameter=False, save_path="./", interval_save_parameter=25)
