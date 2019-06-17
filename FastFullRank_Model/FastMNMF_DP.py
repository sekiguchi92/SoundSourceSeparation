#! /usr/bin/env python3
# coding: utf-8

import sys, os
import numpy as np
import chainer
from chainer import functions as chf
import time
import pickle as pic

from configure import *
from FastFCA import FastFCA

class FastMSDSP(FastFCA):

    def __init__(self, speech_VAE=None, NUM_noise=1, NUM_Z_iteration=30, DIM_latent=16, NUM_basis_noise=2, xp=np, MODE_initialize_covarianceMatrix="unit", MODE_update_Z="sampling", normalize_input=True):
        """ initialize FastMSDSP

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
        super(FastMSDSP, self).__init__(NUM_source=NUM_noise+1, xp=xp, MODE_initialize_covarianceMatrix=MODE_initialize_covarianceMatrix)
        self.NUM_source, self.NUM_speech, self.NUM_noise = NUM_noise+1, 1, NUM_noise
        self.speech_VAE = speech_VAE
        self.NUM_Z_iteration = NUM_Z_iteration
        self.NUM_basis_noise = NUM_basis_noise
        self.DIM_latent = DIM_latent
        self.MODE_update_Z = MODE_update_Z
        self.normalize_input = normalize_input
        self.method_name = "FastMSDSP"
        self.time_for_Z = 0.0
        self.time_for_WH = 0.0
        self.time_for_UV = 0.0
        self.time_for_QG = 0.0


    def load_spectrogram(self, X_FTM):
        super(FastMSDSP, self).load_spectrogram(X_FTM)
        # self.NUM_freq, self.NUM_time, self.NUM_mic = X_FTM.shape
        # self.X_FTM = self.xp.asarray(X_FTM, dtype=self.xp.complex)
        # self.XX_FTMM = self.xp.matmul( self.X_FTM[:, :, :, None], self.xp.conj( self.X_FTM[:, :, None, :] ) ) # F T M M
        # self.lambda_NFT = self.xp.random.random([self.NUM_source, self.NUM_freq, self.NUM_time]).astype(self.xp.float)
        # self.covarianceDiag_NFM = self.xp.ones([self.NUM_source, self.NUM_freq, self.NUM_mic], dtype=self.xp.float) / self.NUM_mic
        # self.diagonalizer_FMM = self.xp.zeros([self.NUM_freq, self.NUM_mic, self.NUM_mic], dtype=self.xp.complex)
        # self.diagonalizer_FMM[:] = self.xp.eye(self.NUM_mic).astype(self.xp.complex)

        self.xp = self.speech_VAE.xp
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
            MODE_initialize_covarianceMatrix: str
                how to initialize covariance matrix {unit, obs, cGMM}
            MODE_update_variable: str
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
        if MODE_update_Z != None:
            self.MODE_update_Z = MODE_update_Z


    def initialize_PSD(self):
        power_observation_FT = (self.xp.abs(self.X_FTM) ** 2).mean(axis=2)
        shape = 2
        self.W_noise_NnFK = self.xp.random.dirichlet(np.ones(self.NUM_freq)*shape, size=[self.NUM_noise, self.NUM_basis_noise]).transpose(0, 2, 1)

        self.H_noise_NnKT[:] = self.xp.random.gamma(shape, (power_observation_FT.mean() * self.NUM_freq * self.NUM_mic / (self.NUM_noise * self.NUM_basis_noise)) / shape, size=[self.NUM_noise, self.NUM_basis_noise, self.NUM_time])
        self.H_noise_NnKT[self.H_noise_NnKT < EPS] = EPS

        self.u_F[:] = 1 / self.NUM_freq
        self.v_T[:] = 1

        if self.normalize_input:
            power_observation_FT = power_observation_FT / power_observation_FT.sum(axis=0).mean()

        self.Z_speech_DT = self.speech_VAE.encode_cupy(power_observation_FT.astype(self.xp.float32))
        self.z_link_speech = Z_link(self.Z_speech_DT.T)
        self.z_optimizer_speech = chainer.optimizers.Adam().setup(self.z_link_speech)
        self.power_speech_FT = self.speech_VAE.decode_cupy(self.Z_speech_DT)

        self.lambda_NFT[0] = self.u_F[:, None] * self.v_T[None] * self.power_speech_FT
        self.lambda_NFT[1:] = self.W_noise_NnFK @ self.H_noise_NnKT
        self.reset_variable()


    def make_filename_suffix(self):
        self.filename_suffix = "N={}-it={}-itZ={}-speech=VAE-Ls=NONE-noise=NMF-Ln={}-D={}-init={}-latent={}".format(self.NUM_noise, self.NUM_iteration, self.NUM_Z_iteration, self.NUM_basis_noise, self.DIM_latent, self.MODE_initialize_covarianceMatrix, self.MODE_update_Z)

        if hasattr(self, "name_DNN"):
            self.filename_suffix += "-DNN={}".format(self.name_DNN)

        if hasattr(self, "file_id"):
            self.filename_suffix += "-ID={}".format(self.file_id)
        else:
            print("====================\n\nWarning: Please set self.file_id\n\n====================")

        print("filename_suffix:", self.filename_suffix)

    # @profile
    # def update(self):
    #     self.update_UV()
    #     self.update_Z_speech()
    #     self.update_WH_noise()
    #     self.update_CovarianceDiagElement()
    #     self.udpate_Diagonalizer()
    #     self.normalize()

    def update(self):
        start = time.time()
        self.update_UV()
        self.time_for_UV += time.time() - start

        start = time.time()
        self.update_Z_speech()
        self.time_for_Z += time.time() - start

        start = time.time()
        self.update_WH_noise()
        self.time_for_WH += time.time() - start

        start = time.time()
        self.update_CovarianceDiagElement()
        self.udpate_Diagonalizer()
        self.normalize()
        self.time_for_QG += time.time() - start


    def normalize(self):
        phi_F = self.xp.trace(self.diagonalizer_FMM @ self.diagonalizer_FMM.conj().transpose(0, 2, 1), axis1=1, axis2=2).real / self.NUM_mic
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
        if self.xp == np:
            for f in range(self.NUM_freq):
                a_1 = self.xp.zeros([self.NUM_noise, self.NUM_basis_noise])
                b_1 = self.xp.zeros([self.NUM_noise, self.NUM_basis_noise])
                for m in range(self.NUM_mic):
                    a_1 += (self.H_noise_NnKT * (self.covarianceDiag_NFM[1:, f, None, m] * (self.Qx_power_FTM[f, :, m] / (self.Y_FTM[f, :, m] ** 2))[None])[:, None]).sum(axis=2)  # N K T
                    b_1 += (self.H_noise_NnKT * (self.covarianceDiag_NFM[1:, f, None, m] / self.Y_FTM[None, f, :, m])[:, None]).sum(axis=2)
                self.W_noise_NnFK[:, f] = self.W_noise_NnFK[:, f] * self.xp.sqrt(a_1 / b_1)

            self.lambda_NFT[1:] = self.W_noise_NnFK @ self.H_noise_NnKT
            self.Y_FTM = (self.lambda_NFT[..., None] * self.covarianceDiag_NFM[:, :, None]).sum(axis=0)

            for t in range(self.NUM_time):
                a_1 = self.xp.zeros([self.NUM_noise, self.NUM_basis_noise])
                b_1 = self.xp.zeros([self.NUM_noise, self.NUM_basis_noise])
                for m in range(self.NUM_mic):
                    a_1 = (self.W_noise_NnFK * (self.covarianceDiag_NFM[1:, :, m] * (self.Qx_power_FTM[:, t, m] / (self.Y_FTM[:, t, m] ** 2))[None])[:, :, None] ).sum(axis=1) # N F K
                    b_1 = (self.W_noise_NnFK * (self.covarianceDiag_NFM[1:, :, m] / self.Y_FTM[None, :, t, m])[:, :, None]).sum(axis=1) # N F K
                self.H_noise_NnKT[:, :, t] = self.H_noise_NnKT[:, :, t] * self.xp.sqrt(a_1 / b_1)

            self.lambda_NFT[1:] = self.W_noise_NnFK @ self.H_noise_NnKT
            self.Y_FTM = (self.lambda_NFT[..., None] * self.covarianceDiag_NFM[:, :, None]).sum(axis=0)

        else:
            a_1 = (self.H_noise_NnKT[:, None, :, :, None] * (self.covarianceDiag_NFM[1:, :, None] * (self.Qx_power_FTM / (self.Y_FTM ** 2))[None])[:, :, None]).sum(axis=4).sum(axis=3)  # N F K T M
            b_1 = (self.H_noise_NnKT[:, None, :, :, None] * (self.covarianceDiag_NFM[1:, :, None] / self.Y_FTM[None])[:, :, None]).sum(axis=4).sum(axis=3)
            self.W_noise_NnFK = self.W_noise_NnFK * self.xp.sqrt(a_1 / b_1)
            self.lambda_NFT[1:] = self.W_noise_NnFK @ self.H_noise_NnKT
            self.Y_FTM = (self.lambda_NFT[..., None] * self.covarianceDiag_NFM[:, :, None]).sum(axis=0)

            a_1 = (self.W_noise_NnFK[..., None, None] * (self.covarianceDiag_NFM[1:, :, None] * (self.Qx_power_FTM / (self.Y_FTM ** 2))[None])[:, :, None] ).sum(axis=4).sum(axis=1) # N F K T M
            b_1 = (self.W_noise_NnFK[..., None, None] * (self.covarianceDiag_NFM[1:, :, None] / self.Y_FTM[None])[:, :, None]).sum(axis=4).sum(axis=1) # N F K T M
            self.H_noise_NnKT = self.H_noise_NnKT * self.xp.sqrt(a_1 / b_1)
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

    # @profile
    def loss_func_Z(self, z, vae, n):
        power_tmp_FT = chf.exp(vae.decode(z).T) + EPS
        Y_tmp_FTM = power_tmp_FT[:, :, None] * self.UVG_FTM+ self.WHG_noise_FTM
        return chf.sum(chf.log(Y_tmp_FTM) + self.Qx_power_FTM / Y_tmp_FTM ) / (self.NUM_freq * self.NUM_mic)

    # @profile
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

            # lambda_old_FTM = power_old_FTM * self.UVG_FTM + self.WHG_noise_FTM
            # for it in range(self.NUM_Z_iteration):
            #     Z_speech_new_DT = chf.gaussian(Z_speech_old_DT, log_var).data
            #     power_new_FTM = self.speech_VAE.decode_cupy(Z_speech_new_DT)
            #     lambda_new_FTM = power_new_FTM[:, :, None] * self.UVG_FTM + self.WHG_noise_FTM
            #     # acceptance_rate = self.xp.exp((self.Qx_power_FTM * (1 / lambda_old_FTM - 1 / lambda_new_FTM) + self.xp.log( lambda_old_FTM /  lambda_new_FTM)).sum(axis=2).sum(axis=0) + (Z_speech_old_DT ** 2 - Z_speech_new_DT ** 2).sum(axis=0) / 2 )
            #     acceptance_rate = self.xp.exp((self.Qx_power_FTM * (1 / lambda_old_FTM - 1 / lambda_new_FTM) + self.xp.log( lambda_old_FTM /  lambda_new_FTM)).sum(axis=2).sum(axis=0) )
            #     acceptance_boolean = self.xp.random.random([self.NUM_time]) < acceptance_rate
            #     Z_speech_old_DT[:, acceptance_boolean] = Z_speech_new_DT[:, acceptance_boolean]
            #     lambda_old_FTM[:, acceptance_boolean] = lambda_new_FTM[:, acceptance_boolean]
            for it in range(self.NUM_Z_iteration):
                Z_speech_new_DT = chf.gaussian(Z_speech_old_DT, log_var).data
                lambda_old_FTM = power_old_FTM * self.UVG_FTM + self.WHG_noise_FTM
                power_new_FTM = self.speech_VAE.decode_cupy(Z_speech_new_DT)[:, :, None]
                lambda_new_FTM = power_new_FTM * self.UVG_FTM + self.WHG_noise_FTM

                # acceptance_rate = self.xp.exp((self.Qx_power_FTM * (1 / lambda_old_FTM - 1 / lambda_new_FTM)).sum(axis=2).sum(axis=0) + self.xp.log( ( lambda_old_FTM / lambda_new_FTM ).prod(axis=2).prod(axis=0) ) + (Z_speech_old_DT ** 2 - Z_speech_new_DT ** 2).sum(axis=0) / 2)
                acceptance_rate = self.xp.exp((self.Qx_power_FTM * (1 / lambda_old_FTM - 1 / lambda_new_FTM)).sum(axis=2).sum(axis=0) + self.xp.log( ( lambda_old_FTM / lambda_new_FTM ).prod(axis=2).prod(axis=0) ) )
                acceptance_boolean = self.xp.random.random([self.NUM_time]) < acceptance_rate

                # print(Z_speech_new_DT.shape, Z_speech_old_DT.shape, acceptance_boolean.shape)
                Z_speech_old_DT[:, acceptance_boolean] = Z_speech_new_DT[:, acceptance_boolean]
                power_old_FTM[:, acceptance_boolean] = power_new_FTM[:, acceptance_boolean]

            self.Z_speech_DT = Z_speech_old_DT
            self.z_link_speech.z = chainer.Parameter(self.Z_speech_DT.T)
            self.power_speech_FT = self.speech_VAE.decode_cupy(self.Z_speech_DT)

        self.lambda_NFT[0] = self.u_F[:, None] * self.v_T[None] * self.power_speech_FT
        self.Y_FTM = (self.lambda_NFT[..., None] * self.covarianceDiag_NFM[:, :, None]).sum(axis=0)


    def save_parameter(self, filename):
        param_list = [self.lambda_NFT, self.covarianceDiag_NFM, self.diagonalizer_FMM, self.u_F, self.v_T, self.Z_speech_DT, self.W_noise_NnFK, self.H_noise_NnKT]
        if self.xp != np:
            param_list = [chainer.cuda.to_cpu(param) for param in param_list]

        pic.dump(param_list, open(filename, "wb"))


    def load_parameter(self, filename):
        param_list = pic.load(open(filename, "rb"))
        if self.xp != np:
            param_list = [chainer.cuda.to_gpu(param) for param in param_list]

        self.lambda_NFT, self.covarianceDiag_NFM, self.diagonalizer_FMM, self.u_F, self.v_T, self.Z_speech_DT, self.W_noise_NnFK, self.H_noise_NnKT = param_list


class Z_link(chainer.link.Link):
    def __init__(self, z):
        super(Z_link, self).__init__()

        with self.init_scope():
            self.z = chainer.Parameter(z)


if __name__ == "__main__":
    import soundfile as sf
    import librosa
    import argparse
    import pickle as pic
    import sys, os
    from chainer import serializers

    sys.path.append("/home/sekiguch/Dropbox/program/python/chainer/")
    sys.path.append("../chainer/")

    parser = argparse.ArgumentParser()
    parser.add_argument(             '--gpu', type=  int, default=     0, help='GPU ID')##

    parser.add_argument(      '--DIM_latent', type=  int, default=   16, help='dimention of encoded vector')
    parser.add_argument(       '--layer_enc', type=  int, default=    1, help='number of layer of encoder')
    parser.add_argument(       '--layer_dec', type=  int, default=    1, help='number of layer of decoder')
    parser.add_argument(       '--NUM_noise', type=  int, default=    1, help='number of noise')
    parser.add_argument(   '--NUM_iteration', type=  int, default=  100, help='number of iteration')
    parser.add_argument( '--NUM_Z_iteration', type=  int, default=   30, help='number of update Z iteration')
    parser.add_argument( '--NUM_basis_noise', type=  int, default=   64, help='number of basis of noise (MODE_noise=NMF)')
    parser.add_argument(   '--MODE_update_Z', type=  str, default="sampling", help='sampling, sampling2, backprop, backprop2, hybrid, hybrid2')
    parser.add_argument( '--MODE_initialize_covarianceMatrix', type=  str, default="obs", help='cGMM, cGMM2, unit, obs')
    args = parser.parse_args()

    if args.gpu < 0:
        import numpy as xp
    else:
        import cupy as xp
        print("Use GPU " + str(args.gpu))
        chainer.cuda.get_device_from_id(args.gpu).use()

    import network_normal
    # data_dir = "/n/sd2/sekiguchi/data_for_chainer/network_normal/"
    data_dir = "./"
    filename_suffix = "normal-scale={}-speech_only=False-input={}-mode=log_power_IS-D={}-layer_enc={}-dec={}.npz".format("gamma", "power", args.DIM_latent, args.layer_enc, args.layer_dec)
    speech_VAE = network_normal.VAE(layer_enc=args.layer_enc, layer_dec=args.layer_dec, n_latent=args.DIM_latent)
    serializers.load_npz(data_dir + "model-best-"+filename_suffix, speech_VAE) # 設定をspeech_vaeにロード
    name_DNN = "gamma_speech_only=False-enc={}_dec={}_IS".format(args.layer_enc, args.layer_dec)

    if xp != np:
        speech_VAE.to_gpu()

    filename = "../../data/chime/F04_050C0115_CAF.CH13456.wav"
    # filename = "/n/sd2/sekiguchi/CHiME3/data/et05/simu/F05_447C020Q_BUS.CH13456.wav"
    wav, fs = sf.read(filename)
    wav = wav.T
    M = len(wav)
    for m in range(M):
        tmp = librosa.core.stft(wav[m], n_fft=1024, hop_length=256)
        if m == 0:
            spec = np.zeros([tmp.shape[0], tmp.shape[1], M], dtype=np.complex)
        spec[:, :, m] = tmp

    separater = FastMSDSP(NUM_noise=args.NUM_noise, speech_VAE=speech_VAE, NUM_Z_iteration=args.NUM_Z_iteration, NUM_basis_noise=args.NUM_basis_noise, xp=xp, MODE_initialize_covarianceMatrix=args.MODE_initialize_covarianceMatrix, MODE_update_Z=args.MODE_update_Z)
    separater.load_spectrogram(spec)

    separater.file_id = "F04_050C0115_CAF"
    separater.name_DNN = name_DNN

    processingTime = separater.solve(NUM_iteration=args.NUM_iteration, save_likelihood=False, save_parameter=False, save_dir="./", interval_save_parameter=300)
    print("processingTime : ", processingTime / args.NUM_iteration)
    # separater.check_processing_time(NUM_Z_iteration=args.NUM_Z_iteration, MODE_update_Z=args.MODE_update_Z, iteration=100)

    clean_filename = "../../data/chime/clean_F04_050C0115_CAF.CH5.wav"
    separater.wav_org = sf.read(clean_filename)[0]
    # separater.separate_FastWienerFilter()
    print("SDR = ", separater.calculate_separation_performance(), "  processing_time:", processingTime/args.NUM_iteration)
    print("UV:", separater.time_for_UV, "Z:", separater.time_for_Z, "WH:", separater.time_for_WH, "QG:", separater.time_for_QG)
