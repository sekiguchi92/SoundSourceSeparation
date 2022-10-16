#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import sys, os
import soundfile as sf

import torch
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path

sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))

from Base import EPS, MIC_INDEX, Base, MultiSTFT


class FastBSSD(Base):
    """
    Joint sound source separation and dereverberation method with various source models
    and weight-shared jointly-diagonalizable (WJD) spatial model.

    You can specify the speech and noise models separately.
    The speech model is 'NMF', 'FreqInv', or 'DNN',
    and the noise model is 'TimeInv' or 'NMF'.
    For the DNN-based speech model, the initialization is important.
    Please use 'init_SCM = twostep', which first run FastBSSD with FreqInv and TimeInv
    and then run FastBSSD with DNN by using the variables estimated by the first FastBSSD.

    For updating the spatial model, you can select the algorithm from 'IP', 'ISS1', and 'ISS2'.
    ISS2 is recommended. ISS1 is computationally inexpensive, but low performance.
    IP performs as well as ISS2 but it is a bit computationally heavier than ISS2.

    Meaning of the variables:
        F: n_freq = the number of frequency bins
        T: n_time = the number of time frames
        M: n_mic =  the number of microphones
        Ns: n_speech = the number of speakers
        Nn: n_noise = the number of noises
        N: n_speech + n_noise
        Lm: the tap length for the MA model
        La: the tap length for the AR model

        X_FTM: the observed multichannel complex spectrogram
        PSD_NFT: power spectral densities (PSD) of each source
        Q_FMM: diagonalizer that converts spatial covariance matrices (SCM) to diagonal matrices
        G_NLmM: diagonal elements of the diagonalized SCMs
        P_FxMxMLa: matrix consisting of AR filters and diagonalizer (F x M x M*La)
        Px_power_FTM: power spectra of Px
        Y_FTM: sum_n PSD_NFT G_NLmM

        For NMF source model:
            Ks: the number of bases of NMF for speech
            Kn: the number of bases of NMF for noise

            W_NsFKs: basis vectors for speech source
            H_NsKsT: activations for speech source
            W_noise_NnFKn: basis vectors for noise source
            H_noise_NnKnT: activations for noise source

        For FreqInv source model:
            PSD_NsT: PSD for speech source

        For DNN speech model:
            U_NsF: parameter for normalization
            V_NsT: time activation
            Z_NsDT: the latent variable
            power_speech_NsxFxT: the output of VAE decoder given Z_NsDT
    """

    method_name = "FastBSSD"

    def __init__(
        self,
        n_speech=2,
        n_noise=0,
        speech_model=["NMF", "FreqInv", "DNN"][0],
        noise_model=["TimeInv", "NMF"][0],
        algo=["IP", "ISS1", "ISS2"][0],
        xp=np,
        init_SCM="circular",
        n_tap_AR=3,
        n_tap_MA=2,
        n_delay_AR=3,
        n_tap_direct=0,
        n_basis=8,
        n_basis_noise=8,
        speech_VAE=None,
        n_iter_z=10,
        lr=1e-3,
        n_bit=64,
        g_eps=1e-2,
        n_iter_init=30,
        interval_norm=10,
        **kwargs,
    ):
        """initialize FastBSSD

        Parameters:
        -----------
            n_speech: int
                The number of sources
            n_noise: int
                The number of noise.
            speech_model: str
                NMF, FreqInv, or DNN
            noise_model: str
                TimeInv, or NMF
            algo: str
                The algorithm for updating Q (IP or ISS)
            xp: numpy or cupy
            init_SCM: str
                The method for initializing the SCMs (circular, obs, or twostep)
            n_tap_MA: int
                Tap length for early reflection
            n_tap_AR: int
                Tap length for late reverberation
            n_delay: int
                The index to indicate the beginning of the late reverberation ( > n_tap_MA )

            # Only for NMF source model
            n_basis, n_basis_noise: int
                The number of bases for NMF source/noise model

            # Only for DNN source model
            speech_VAE: torch.nn.Module
                The variational autoencoder trained with clean speech signals
            n_iter_z: int
                The number of iterations for updating Z in one iteration
            lr: float
                The learning rate for updating Z

            # Other parameters
            g_eps: float
                The initial value of non-diagonal elements of G_NM
            n_iter_init: str
                The number of iterations used when init_SCM = twostep
            interval_norm: int
                Interval of normalization
        """
        super(FastBSSD, self).__init__(xp=xp, n_bit=n_bit)
        self.n_speech = n_speech
        self.n_noise = n_noise
        self.n_source = self.n_speech + self.n_noise
        self.speech_model = speech_model
        self.noise_model = noise_model if self.n_noise > 0 else None
        self.algo = algo
        self.init_SCM = init_SCM
        self.n_tap_AR = n_tap_AR
        self.n_tap_MA = n_tap_MA
        self.n_delay_AR = n_delay_AR
        self.n_tap_direct = n_tap_direct

        if self.n_tap_MA == 0:
            self.n_tap_direct = 0
        if self.n_tap_AR == 0:
            self.n_delay_AR = 0

        self.n_basis = n_basis
        self.n_basis_noise = n_basis_noise

        self.speech_VAE = speech_VAE
        self.n_iter_z = n_iter_z
        self.lr = lr

        self.g_eps = g_eps
        self.n_iter_init = n_iter_init
        self.interval_norm = interval_norm

        if self.speech_model == "NMF":
            self.save_param_list += ["W_NsFKs", "H_NsKsT"]
        elif self.speech_model == "FreqInv":
            self.save_param_list += ["PSD_NsT"]
        elif self.speech_model == "DNN":
            self.save_param_list += ["U_NsF", "V_NsT", "Z_NsDT"]

        if self.n_noise > 0:
            self.save_param_list += ["W_noise_NnFKn", "H_noise_NnKnT"]

        self.method_name = "FastBSSD"

    def __str__(self):
        if self.speech_model == "NMF":
            speech_model_name = f"NMF_K{self.n_basis}"
        elif self.speech_model == "DNN":
            speech_model_name = f"DNN_it{self.n_iter_z}"
        else:
            speech_model_name = self.speech_model

        if self.n_noise > 0:
            noise_model_name = f"NMF_K{self.n_basis_noise}" if "NMF" == self.noise_model else self.noise_model
        else:
            noise_model_name = "None"

        init_SCM_name = f"twostep_it{self.n_iter_init}" if self.init_SCM == "twostep" else self.init_SCM

        modelname = (
            f"speech={speech_model_name}-noise={noise_model_name}-M={self.n_mic}"
            f"-Ns={self.n_speech}-Nn={self.n_noise}-it={self.n_iter}-init={init_SCM_name}"
            f"-Delay={self.n_delay_AR}-L_AR={self.n_tap_AR}-L_MA={self.n_tap_MA}"
        )

        if self.n_bit == 32:
            modelname += "-bit=32"
        if hasattr(self, "file_id"):
            modelname += f"-ID={self.file_id}"
        return modelname

    def calculate_log_likelihood(self):
        self.calculate_PSD()
        self.calculate_Px()
        self.calculate_Y()
        self.calculate_Px_power()
        self.log_likelihood = (
            -(self.Px_power_FTM / self.Y_FTM + self.xp.log(self.Y_FTM)).sum()
            + self.n_time
            * (self.xp.log(self.xp.linalg.det(self.Q_FMM @ self.Q_FMM.transpose(0, 2, 1).conj()).real)).sum()
        )
        return self.log_likelihood

    def load_spectrogram(self, X_FTM, sample_rate=16000):
        super().load_spectrogram(X_FTM)

        self.Xbar_FxTxMLa = self.xp.zeros(
            [self.n_freq, self.n_time, self.n_mic * (self.n_tap_AR + 1)], dtype=self.TYPE_COMPLEX
        )
        self.Xbar_FxTxMLa[:, :, : self.n_mic] = self.X_FTM
        for l in range(self.n_tap_AR):
            self.Xbar_FxTxMLa[:, self.n_delay_AR + l :, (l + 1) * self.n_mic : (l + 2) * self.n_mic] = self.X_FTM[
                :, : -(self.n_delay_AR + l)
            ]
        self.Px_FTM = self.X_FTM.copy()

    def init_source_model(self):
        if self.speech_model == "NMF":
            self.W_NsFKs = self.xp.random.rand(self.n_speech, self.n_freq, self.n_basis).astype(self.TYPE_FLOAT)
            self.H_NsKsT = self.xp.random.rand(self.n_speech, self.n_basis, self.n_time).astype(self.TYPE_FLOAT)
        elif self.speech_model == "FreqInv":
            self.PSD_NsT = self.xp.random.rand(self.n_speech, self.n_time).astype(self.TYPE_FLOAT) + EPS
        elif self.speech_model == "DNN":
            self.U_NsF = self.xp.ones([self.n_speech, self.n_freq])
            self.V_NsT = self.xp.ones([self.n_speech, self.n_time])
            self.torch_device = "cpu" if self.xp is np else f"cuda:{self.X_FTM.device.id}"

        if self.n_noise > 0:
            if self.noise_model == "NMF":
                self.W_noise_NnFKn = self.xp.random.rand(self.n_noise, self.n_freq, self.n_basis).astype(
                    self.TYPE_FLOAT
                )
                self.H_noise_NnKnT = self.xp.random.rand(self.n_noise, self.n_basis, self.n_time).astype(
                    self.TYPE_FLOAT
                )
            elif self.noise_model == "TimeInv":
                self.W_noise_NnFKn = self.xp.ones([self.n_noise, self.n_freq, 1]).astype(self.TYPE_FLOAT)
                self.H_noise_NnKnT = self.xp.ones([self.n_noise, 1, self.n_time]).astype(self.TYPE_FLOAT)
        self.PSD_NFT = self.xp.zeros([self.n_source, self.n_freq, self.n_time], dtype=self.TYPE_FLOAT)

        self.P_FxMxMLa = self.xp.zeros(
            [self.n_freq, self.n_mic, self.n_mic * (self.n_tap_AR + 1)], dtype=self.TYPE_COMPLEX
        )
        self.Px_FTM = self.X_FTM.copy()

    def init_spatial_model(self):
        self.start_idx = 0
        self.Q_FMM = self.xp.tile(self.xp.eye(self.n_mic), [self.n_freq, 1, 1]).astype(self.TYPE_COMPLEX)
        self.P_FxMxMLa[:, :, : self.n_mic] = self.Q_FMM
        self.G_NLmM = self.g_eps * self.xp.ones([self.n_source, self.n_tap_MA + 1, self.n_mic], dtype=self.TYPE_FLOAT)

        if "circular" in self.init_SCM:
            for m in range(self.n_mic):
                self.G_NLmM[m % self.n_source, 0, m] = 1

        elif "twostep" in self.init_SCM:
            if self.n_iter_init >= self.n_iter:
                print(
                    "\n------------------------------------------------------------------\n"
                    f"Warning: n_iter_init must be smaller than n_iter (= {self.n_iter}).\n"
                    f"n_iter_init is changed from {self.n_iter_init} to {self.n_iter // 3}"
                    "\n------------------------------------------------------------------\n"
                )
                self.n_iter_init = self.n_iter // 3

            self.start_idx = self.n_iter_init

            separater_init = FastBSSD(
                n_speech=self.n_speech,
                n_noise=self.n_noise,
                speech_model="FreqInv",
                noise_model="TimeInv",
                init_SCM="circular",
                xp=self.xp,
                n_bit=self.n_bit,
                n_tap_MA=0,
                n_tap_AR=self.n_tap_AR,
                n_delay_AR=self.n_delay_AR,
                g_eps=self.g_eps,
            )
            separater_init.file_id = self.file_id
            separater_init.load_spectrogram(self.X_FTM)
            separater_init.solve(n_iter=self.start_idx, save_wav=False)
            self.P_FxMxMLa = separater_init.P_FxMxMLa
            self.Q_FMM = self.P_FxMxMLa[:, :, : self.n_mic]

            self.G_NLmM = self.g_eps * self.xp.ones(
                [self.n_source, self.n_tap_MA + 1, self.n_mic], dtype=self.TYPE_FLOAT
            )
            self.G_NLmM[:, 0] = separater_init.G_NLmM[:, 0]

            if self.speech_model == "DNN":
                power_speech_NsxFxT = self.xp.asarray(
                    np.abs(separater_init.separated_spec[: self.n_speech]) ** 2
                ).astype(self.xp.float32)
                power_speech_NsxFxT /= power_speech_NsxFxT.sum(axis=1).mean(axis=1)[:, None, None]
                with torch.set_grad_enabled(False):
                    self.Z_NsDT = self.speech_VAE.encode_(
                        torch.as_tensor(power_speech_NsxFxT + EPS, device=self.torch_device)
                    ).detach()
                    self.Z_NsDT.requires_grad = True
                    self.z_optimizer = optim.AdamW([self.Z_NsDT], lr=self.lr)
                    self.power_speech_NsxFxT = self.xp.asarray(self.speech_VAE.decode_(self.Z_NsDT))

        else:
            print(f"Please specify how to initialize covariance matrix {separater.init_SCM}")
            raise ValueError

        self.P_FxMxMLa[:, :, self.n_mic :] = 0
        self.G_NLmM /= self.G_NLmM.sum(axis=(1, 2))[:, None, None]
        self.normalize()

    def reset_variable(self):
        self.calculate_Px()
        self.calculate_Px_power()
        self.calculate_Y()

    def calculate_PSD(self):
        if self.speech_model == "NMF":
            self.PSD_NFT[: self.n_speech] = self.W_NsFKs @ self.H_NsKsT + EPS
        elif self.speech_model == "FreqInv":
            self.PSD_NFT[: self.n_speech] = self.PSD_NsT[:, None]
        elif self.speech_model == "DNN":
            self.PSD_NFT[: self.n_speech] = self.U_NsF[:, :, None] * self.V_NsT[:, None] * self.power_speech_NsxFxT

        if self.n_noise > 0:
            self.PSD_NFT[self.n_speech :] = self.W_noise_NnFKn @ self.H_noise_NnKnT

    def calculate_Px(self):
        self.Px_FTM = (self.P_FxMxMLa[:, None] * self.Xbar_FxTxMLa[:, :, None]).sum(axis=3)

    def calculate_Px_power(self):
        self.Px_power_FTM = self.xp.abs(self.Px_FTM) ** 2

    def calculate_Yn(self):
        self.calculate_PSD()
        self.Y_NFTM = self.PSD_NFT[:, :, :, None] * self.G_NLmM[:, 0, None, None]
        if self.n_tap_direct > 0:
            for l in range(1, self.n_tap_direct + 1):
                self.Y_NFTM[:, :, l:] += self.PSD_NFT[:, :, :-l, None] * self.G_NLmM[:, l, None, None]

    def calculate_Y(self):
        self.calculate_PSD()
        self.Y_FTM = (self.PSD_NFT[:, :, :, None] * self.G_NLmM[:, 0, None, None]).sum(axis=0)
        for l in range(1, 1 + self.n_tap_MA):
            self.Y_FTM[:, l:] += (self.PSD_NFT[:, :, :-l, None] * self.G_NLmM[:, l, None, None]).sum(axis=0)
        self.Y_FTM += EPS

    def update(self):
        self.update_PSD()
        self.update_G()
        self.update_AR()

        if self.it % self.interval_norm == 0:
            self.normalize()

    def update_PSD(self):
        if self.speech_model == "NMF":
            self.update_PSD_NMF()
        elif self.speech_model == "FreqInv":
            self.update_PSD_FreqInv()
        elif self.speech_model == "DNN":
            self.update_PSD_DNN()

        if self.noise_model == "NMF":
            self.update_PSD_NMF_noise()

    def update_PSD_NMF(self):
        XY2_FTM = self.Px_power_FTM / (self.Y_FTM**2)
        HG_NsKLTM = self.H_NsKsT[:, :, None, :, None] * self.G_NLmM[: self.n_speech, None, :, None]
        HG_sum_NsKTM = HG_NsKLTM[:, :, 0].copy()  # copyいらなさそう
        for l in range(1, 1 + self.n_tap_MA):
            HG_sum_NsKTM[:, :, l:] += HG_NsKLTM[:, :, l, :-l]
        a_W_NsFK = (HG_sum_NsKTM[:, None] * XY2_FTM[None, :, None]).sum(axis=(3, 4))
        b_W_NsFK = (HG_sum_NsKTM[:, None] / self.Y_FTM[None, :, None]).sum(axis=(3, 4))

        self.W_NsFKs *= self.xp.sqrt(a_W_NsFK / b_W_NsFK)
        self.calculate_Y()

        XY2_FTM = self.Px_power_FTM / (self.Y_FTM**2)
        WXY2_NsKTM = (self.W_NsFKs[:, :, :, None, None] * XY2_FTM[None, :, None]).sum(axis=1)
        WYinv_NKTM = (self.W_NsFKs[:, :, :, None, None] / self.Y_FTM[None, :, None]).sum(axis=1)
        GWXY2_NsKTM = self.G_NLmM[: self.n_speech, 0, None, None] * WXY2_NsKTM
        GWYinv_NsKTM = self.G_NLmM[: self.n_speech, 0, None, None] * WYinv_NKTM
        for l in range(1, 1 + self.n_tap_MA):
            GWXY2_NsKTM[:, :, :-l] += self.G_NLmM[: self.n_speech, l, None, None] * WXY2_NsKTM[:, :, l:]
            GWYinv_NsKTM[:, :, :-l] += self.G_NLmM[: self.n_speech, l, None, None] * WYinv_NKTM[:, :, l:]
        a_H_NsKT = GWXY2_NsKTM.sum(axis=3)
        b_H_NsKT = GWYinv_NsKTM.sum(axis=3)

        self.H_NsKsT *= self.xp.sqrt(a_H_NsKT / b_H_NsKT)
        self.calculate_Y()

    def update_PSD_NMF_noise(self):
        XY2_FTM = self.Px_power_FTM / (self.Y_FTM**2)
        HG_NnKLTM = self.H_noise_NnKnT[:, :, None, :, None] * self.G_NLmM[self.n_speech :, None, :, None]
        HG_sum_NnKTM = HG_NnKLTM[:, :, 0].copy()  # copyいらなさそう
        for l in range(1, 1 + self.n_tap_MA):
            HG_sum_NnKTM[:, :, l:] += HG_NnKLTM[:, :, l, :-l]
        a_W_NnFK = (HG_sum_NnKTM[:, None] * XY2_FTM[None, :, None]).sum(axis=(3, 4))
        b_W_NnFK = (HG_sum_NnKTM[:, None] / self.Y_FTM[None, :, None]).sum(axis=(3, 4))

        self.W_noise_NnFKn *= self.xp.sqrt(a_W_NnFK / b_W_NnFK)
        self.calculate_Y()

        XY2_FTM = self.Px_power_FTM / (self.Y_FTM**2)
        WXY2_NnKTM = (self.W_noise_NnFKn[:, :, :, None, None] * XY2_FTM[None, :, None]).sum(axis=1)
        WYinv_NKTM = (self.W_noise_NnFKn[:, :, :, None, None] / self.Y_FTM[None, :, None]).sum(axis=1)
        GWXY2_NnKTM = self.G_NLmM[self.n_speech :, 0, None, None] * WXY2_NnKTM
        GWYinv_NnKTM = self.G_NLmM[self.n_speech :, 0, None, None] * WYinv_NKTM
        for l in range(1, 1 + self.n_tap_MA):
            GWXY2_NnKTM[:, :, :-l] += self.G_NLmM[self.n_speech :, l, None, None] * WXY2_NnKTM[:, :, l:]
            GWYinv_NnKTM[:, :, :-l] += self.G_NLmM[self.n_speech :, l, None, None] * WYinv_NKTM[:, :, l:]
        a_H_NnKT = GWXY2_NnKTM.sum(axis=3)
        b_H_NnKT = GWYinv_NnKTM.sum(axis=3)

        self.H_noise_NnKnT *= self.xp.sqrt(a_H_NnKT / b_H_NnKT)
        self.calculate_Y()

    def update_PSD_FreqInv(self):
        XY2_TM = (self.Px_power_FTM / (self.Y_FTM**2)).sum(axis=0)
        GXY2_NsLTM = self.G_NLmM[: self.n_speech, :, None] * XY2_TM[None, None]
        GY_NsLTM = self.G_NLmM[: self.n_speech, :, None] * (1 / self.Y_FTM).sum(axis=0)[None, None]
        GXY2_sum_NsTM = GXY2_NsLTM[:, 0].copy()
        GY_sum_NsTM = GY_NsLTM[:, 0].copy()
        for l in range(1, 1 + self.n_tap_MA):
            GXY2_sum_NsTM[:, :-l] += GXY2_NsLTM[:, l, l:]
            GY_sum_NsTM[:, :-l] += GY_NsLTM[:, l, l:]
        self.PSD_NsT *= self.xp.sqrt(GXY2_sum_NsTM.sum(axis=2) / GY_sum_NsTM.sum(axis=2))
        self.calculate_Y()

    def update_PSD_DNN(self):
        def loss_fn(Y_noise_FTM_torch, G_NLmM_torch, UV_NsFT_torch):  # for update Z by backprop
            power_speech_NsxFxT = self.speech_VAE.decode_(self.Z_NsDT)
            lambda_tmp_NsFT = UV_NsFT_torch * power_speech_NsxFxT  # + EPS
            if self.n_tap_MA > 0:
                lambda_tmp_NsFT = F.pad(lambda_tmp_NsFT, [self.n_tap_MA, 0], mode="constant", value=0)
                Y_tmp_FTM = (
                    (lambda_tmp_NsFT[:, :, self.n_tap_MA :, None] * G_NLmM_torch[: self.n_speech, 0, None, None]).sum(
                        axis=0
                    )
                    + Y_noise_FTM_torch
                    + EPS
                )
                for l in range(1, 1 + self.n_tap_MA):
                    Y_tmp_FTM = Y_tmp_FTM + (
                        lambda_tmp_NsFT[:, :, self.n_tap_MA - l : -l, None]
                        * G_NLmM_torch[: self.n_speech, l, None, None]
                    ).sum(axis=0)
            else:
                Y_tmp_FTM = (
                    (lambda_tmp_NsFT[..., None] * G_NLmM_torch[: self.n_speech, 0, None, None]).sum(axis=0)
                    + Y_noise_FTM_torch
                    + EPS
                )
            return (
                torch.log(Y_tmp_FTM) + torch.as_tensor(self.Px_power_FTM, device=self.torch_device) / Y_tmp_FTM
            ).sum() / (self.n_freq * self.n_mic)

        if self.n_noise > 0:
            Y_noise_FTM_torch = (
                self.PSD_NFT[self.n_speech :, :, :, None] * self.G_NLmM[self.n_speech :, 0, None, None]
            ).sum(axis=0)
            for l in range(1, 1 + self.n_tap_MA):
                Y_noise_FTM_torch[:, l:] += (
                    self.PSD_NFT[self.n_speech :, :, :-l, None] * self.G_NLmM[self.n_speech :, l, None, None]
                ).sum(axis=0)
        else:
            Y_noise_FTM_torch = self.xp.zeros_like(self.X_FTM, dtype=self.TYPE_FLOAT)
        Y_noise_FTM_torch = torch.as_tensor(Y_noise_FTM_torch, device=self.torch_device)
        G_NLmM_torch = torch.as_tensor(self.G_NLmM, device=self.torch_device)
        UV_NsFT_torch = torch.as_tensor(self.U_NsF[:, :, None] * self.V_NsT[:, None], device=self.torch_device)

        for it in range(self.n_iter_z):
            self.z_optimizer.zero_grad()
            loss = loss_fn(Y_noise_FTM_torch, G_NLmM_torch, UV_NsFT_torch)
            loss.backward()
            self.z_optimizer.step()

        with torch.set_grad_enabled(False):
            self.power_speech_NsxFxT = self.xp.asarray(self.speech_VAE.decode_(self.Z_NsDT))
        self.calculate_Y()

        XY2_FTM = self.Px_power_FTM / (self.Y_FTM**2)
        VZG_NsFTLM = (self.V_NsT[:, :, None, None] * self.G_NLmM[: self.n_speech, None])[
            :, None
        ] * self.power_speech_NsxFxT[..., None, None]
        VZG_sum_NsFTM = VZG_NsFTLM[:, :, :, 0]
        for l in range(1, 1 + self.n_tap_MA):
            VZG_sum_NsFTM[:, :, l:] += VZG_NsFTLM[:, :, :-l, l]
        a_U_NsF = (VZG_sum_NsFTM * XY2_FTM[None]).sum(axis=(2, 3))
        b_U_NsF = (VZG_sum_NsFTM / self.Y_FTM[None]).sum(axis=(2, 3))

        self.U_NsF *= self.xp.sqrt(a_U_NsF / b_U_NsF)
        self.calculate_Y()

        XY2_FTM = self.Px_power_FTM / (self.Y_FTM**2)
        GXY2_NFTL = (self.G_NLmM[: self.n_speech, None, None] * XY2_FTM[None, :, :, None]).sum(axis=-1)
        GYinv_NFTL = (self.G_NLmM[: self.n_speech, None, None] / self.Y_FTM[None, :, :, None]).sum(axis=-1)
        GXY2_sum_NFT = GXY2_NFTL[..., 0].copy()
        GYinv_sum_NFT = GYinv_NFTL[..., 0].copy()
        for l in range(1, 1 + self.n_tap_MA):
            GXY2_sum_NFT[:, :, :-l] += GXY2_NFTL[:, :, l:, l]
            GYinv_sum_NFT[:, :, :-l] += GYinv_NFTL[:, :, l:, l]
        a_V_NsT = ((self.U_NsF[:, :, None] * self.power_speech_NsxFxT) * GXY2_sum_NFT).sum(axis=1)
        b_V_NsT = ((self.U_NsF[:, :, None] * self.power_speech_NsxFxT) * GYinv_sum_NFT).sum(axis=1)

        self.V_NsT *= self.xp.sqrt(a_V_NsT / b_V_NsT)
        self.calculate_Y()

    def update_G(self):
        XY2_FTM = self.Px_power_FTM / (self.Y_FTM**2)
        a_G_NM = ((self.PSD_NFT)[..., None] * XY2_FTM[None]).sum(axis=(1, 2))
        b_G_NM = ((self.PSD_NFT)[..., None] / self.Y_FTM[None]).sum(axis=(1, 2))
        self.G_NLmM[:, 0] *= self.xp.sqrt(a_G_NM / b_G_NM)
        for l in range(1, 1 + self.n_tap_MA):
            a_G_NM = ((self.PSD_NFT[:, :, :-l])[..., None] * XY2_FTM[None, :, l:]).sum(axis=(1, 2))
            b_G_NM = ((self.PSD_NFT[:, :, :-l])[..., None] / self.Y_FTM[None, :, l:]).sum(axis=(1, 2))
            self.G_NLmM[:, l] *= self.xp.sqrt(a_G_NM / b_G_NM)
        self.calculate_Y()

    def update_AR(self):
        if self.algo == "IP":
            for m in range(self.n_mic):
                Vinv_FxMLaxMLa = self.xp.linalg.inv(
                    self.xp.einsum(
                        "fti, ftj -> fij", self.Xbar_FxTxMLa, self.Xbar_FxTxMLa.conj() / self.Y_FTM[:, :, m, None]
                    )
                    / self.n_time
                )
                u_FM = self.xp.linalg.inv(self.P_FxMxMLa[:, :, : self.n_mic])[:, :, m]
                self.P_FxMxMLa[:, m] = (
                    (Vinv_FxMLaxMLa[:, :, : self.n_mic] * u_FM[:, None]).sum(axis=2)
                    / self.xp.sqrt(
                        (u_FM.conj() * (Vinv_FxMLaxMLa[:, : self.n_mic, : self.n_mic] * u_FM[:, None]).sum(axis=2))
                        .sum(axis=1)
                        .real
                    )[:, None]
                ).conj()
            self.calculate_Px()

        elif "ISS" in self.algo:
            for m in range(self.n_mic):
                QdQd_FTM = self.Px_FTM * self.Px_FTM[:, :, m, None].conj()
                V_tmp_FxM = (QdQd_FTM[:, :, m, None] / self.Y_FTM).mean(axis=1)
                V_FxM = (QdQd_FTM / self.Y_FTM).mean(axis=1) / V_tmp_FxM
                V_FxM[:, m] = 1 - 1 / self.xp.sqrt(V_tmp_FxM[:, m])
                self.Px_FTM -= self.xp.einsum("fm, ft -> ftm", V_FxM, self.Px_FTM[:, :, m])
                self.P_FxMxMLa -= self.xp.einsum("fi, fj -> fij", V_FxM, self.P_FxMxMLa[:, m])

            if self.n_tap_AR > 0:
                if self.algo == "ISS1":
                    for m in range(self.n_mic, (self.n_tap_AR + 1) * self.n_mic):
                        a_FxM = ((self.Px_FTM / self.Y_FTM) * self.Xbar_FxTxMLa[:, :, m, None].conj()).sum(axis=1)
                        b_FxM = ((self.xp.abs(self.Xbar_FxTxMLa[:, :, m]) ** 2)[:, :, None] / self.Y_FTM).sum(axis=1)
                        V_FxM = a_FxM / b_FxM
                        self.P_FxMxMLa[:, :, m] -= V_FxM
                        self.Px_FTM -= V_FxM[:, None] * self.Xbar_FxTxMLa[:, :, m, None]
                elif self.algo == "ISS2":
                    a_FxMxML = self.xp.einsum(
                        "ftm, fti -> fmi", self.Px_FTM / self.Y_FTM, self.Xbar_FxTxMLa[:, :, self.n_mic :].conj()
                    )
                    c_FxMxML = self.xp.zeros(
                        [self.n_freq, self.n_mic, self.n_mic * self.n_tap_AR], dtype=self.TYPE_COMPLEX
                    )
                    for m in range(self.n_mic):
                        b_FxMLxML = self.xp.linalg.inv(
                            self.xp.einsum(
                                "fti, ftj -> fij",
                                self.Xbar_FxTxMLa[:, :, self.n_mic :] / self.Y_FTM[:, :, m, None],
                                self.Xbar_FxTxMLa[:, :, self.n_mic :].conj(),
                            )
                        )
                        c_FxMxML[:, m] = self.xp.einsum("fi, fij -> fj", a_FxMxML[:, m], b_FxMLxML)
                    self.P_FxMxMLa[:, :, self.n_mic :] -= c_FxMxML
                    self.Px_FTM -= (c_FxMxML[:, None] @ self.Xbar_FxTxMLa[:, :, self.n_mic :, None]).squeeze()

        self.Q_FMM = self.P_FxMxMLa[:, :, : self.n_mic]
        self.calculate_Px_power()

    def normalize(self):
        if self.speech_model in ["NMF", "DNN"]:
            phi_F = self.xp.sum(self.Q_FMM * self.Q_FMM.conj(), axis=(1, 2)).real / self.n_mic
            self.P_FxMxMLa /= self.xp.sqrt(phi_F)[:, None, None]
            if self.speech_model == "NMF":
                self.W_NsFKs /= phi_F[None, :, None]
            elif self.speech_model == "DNN":
                self.U_NsF /= phi_F[None]
            if self.n_noise > 0:
                self.W_noise_NnFKn /= phi_F[None, :, None]

        mu_N = (self.G_NLmM).sum(axis=(1, 2))
        self.G_NLmM /= mu_N[:, None, None]
        if self.speech_model == "NMF":
            self.W_NsFKs *= mu_N[: self.n_speech, None, None]
        elif self.speech_model == "FreqInv":
            self.PSD_NsT *= mu_N[: self.n_speech, None]
        elif self.speech_model == "DNN":
            self.U_NsF *= mu_N[: self.n_speech, None]
        if self.n_noise > 0:
            self.W_noise_NnFKn *= mu_N[self.n_speech :, None, None]

        if self.speech_model == "NMF":
            nu_NsK = self.W_NsFKs.sum(axis=1)
            self.W_NsFKs /= nu_NsK[:, None]
            self.H_NsKsT *= nu_NsK[:, :, None]

        if self.speech_model == "DNN":
            nu_Ns = self.U_NsF.sum(axis=1)
            self.U_NsF /= nu_Ns[:, None]
            self.V_NsT *= nu_Ns[:, None]

        if self.n_noise > 0:
            nu_NnK = self.W_noise_NnFKn.sum(axis=1)
            self.W_noise_NnFKn /= nu_NnK[:, None]
            self.H_noise_NnKnT *= nu_NnK[:, :, None]

        self.reset_variable()

    def separate(self, mic_index=MIC_INDEX):
        self.calculate_Yn()
        self.calculate_Y()
        self.calculate_Px()
        Q_inv_FMM = self.xp.linalg.inv(self.Q_FMM)

        for n in range(self.n_speech):
            tmp = (Q_inv_FMM[:, None, mic_index] * (self.Y_NFTM[n] / self.Y_FTM * self.Px_FTM)).sum(axis=2)
            if n == 0:
                self.separated_spec = np.zeros([self.n_speech, self.n_freq, self.n_time], dtype=np.complex128)
            self.separated_spec[n] = self.convert_to_NumpyArray(tmp)
        return self.separated_spec

    def load_param(self, filename):
        super().load_param(filename)

        if hasattr(self, "W_NsFKs"):
            self.speech_model = "NMF"
            self.n_speech, _, self.n_basis = self.W_NsFKs.shape
        elif hasattr(self, "PSD_NsT"):
            self.speech_model = "FreqInv"
            self.n_speech = self.PSD_NsT.shape[0]
        elif hasattr(self, "U_NsF"):
            self.speech_model = "DNN"
            self.n_speech = self.U_NsF.shape[0]

        if hasattr(self, "W_noise_NnFKn"):
            self.n_noise, _, self.n_basis = self.W_noise_NnFKn.shape
            if self.n_basis == 1:
                self.noise_model = "TimeInv"
            else:
                self.noise_model = "NMF"

        self.n_tap_MA = self.G_NLmM[1] - 1
        self.n_tap_AR = (self.P_FxMxMLa.shape[2] / self.n_mic) - 1


if __name__ == "__main__":
    import argparse
    import pickle as pic
    import sys, os

    parser = argparse.ArgumentParser()
    parser.add_argument("input_fname", type=str, help="filename of the multichannel observed signals")

    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--n_speech", type=int, default=3, help="number of speech")
    parser.add_argument("--n_noise", type=int, default=0, help="number of noise")
    parser.add_argument("--speech_model", type=str, default="NMF", help="NMF, FreqInv, DNN")
    parser.add_argument("--noise_model", type=str, default="NMF", help="TimeInv, NMF")
    parser.add_argument("--n_iter", type=int, default=100, help="number of iteration")
    parser.add_argument("--init_SCM", type=str, default="twostep", help="circular or twostep")
    parser.add_argument("--algo", type=str, default="IP", help="IP or ISS")
    parser.add_argument("--n_tap_MA", type=int, default=8, help="filter length for MA model")
    parser.add_argument("--n_tap_AR", type=int, default=4, help="filter length for AR model")
    parser.add_argument("--n_delay_AR", type=int, default=3, help="delay parameter for AR model")

    # Only for NMF source model
    parser.add_argument("--n_basis", type=int, default=16, help="number of basis for speech")
    parser.add_argument("--n_basis_noise", type=int, default=16, help="number of basis for noise")

    # Only for DNN source model
    parser.add_argument("--n_iter_z", type=int, default=10, help="number of iteration for updating Z")

    # Optional parameters
    parser.add_argument("--file_id", type=str, default=None, help="file id used for saving the result")
    parser.add_argument("--n_mic", type=int, default=8, help="number of microphone")
    parser.add_argument("--n_fft", type=int, default=1024, help="number of frequencies")
    parser.add_argument("--n_bit", type=int, default=64, help="number of bit for float and complex")
    parser.add_argument("--g_eps", type=float, default=1e-2, help="initial value of non-diagonal element of G_NM")
    parser.add_argument("--n_iter_init", type=int, default=30, help="number of iteration for twostep initialization")
    parser.add_argument("--interval_norm", type=int, default=10, help="interval of normalization")
    args = parser.parse_args()

    if args.gpu < 0:
        import numpy as xp
    else:
        import cupy as xp

        print("Use GPU " + str(args.gpu))
        xp.cuda.Device(args.gpu).use()

    # wav -> spectrogram
    wav, sample_rate = sf.read(args.input_fname)
    wav /= np.abs(wav).max() * 1.2
    M = min(len(wav), args.n_mic)
    spec_FTM = MultiSTFT(wav[:, :M], n_fft=args.n_fft)

    # Setting for DNN speech model
    if args.speech_model == "DNN":
        nn_root = f"{str(Path(os.path.abspath(__file__)).parents[2])}/nn/"
        sys.path.append(nn_root)
        from VAE_conv1d import VAE

        device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu")
        speech_VAE = VAE(n_freq=args.n_fft // 2 + 1, use_dropout=True, p_dropbout=0.2)
        nn_fname = f"{nn_root}/{speech_VAE.network_name}-{speech_VAE.filename_suffix}-vad=False.pth"
        state_dict = torch.load(nn_fname)
        speech_VAE.load_state_dict(state_dict["net_state_dict"])
        speech_VAE.to(device)

        # The input length must be a multiple of 4
        spec_FTM = spec_FTM[:, : spec_FTM.shape[1] // 4 * 4]
    else:
        speech_VAE = None

    # Start separation
    separater = FastBSSD(speech_VAE=speech_VAE, xp=xp, **vars(args))
    separater.file_id = args.file_id
    separater.load_spectrogram(spec_FTM, sample_rate)
    separater.solve(
        n_iter=args.n_iter,
        save_likelihood=False,
        save_param=False,
        save_wav=True,
        save_dir="./",
        interval_save=100,
    )
