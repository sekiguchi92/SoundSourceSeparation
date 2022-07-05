#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import sys, os
import soundfile as sf

import torch
import torch.optim as optim

from pathlib import Path

sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))
from Base import EPS, MIC_INDEX, Base, MultiSTFT


class FastBSS2(Base):
    """
    Sound source separation method with various source models and
    weight-shared jointly-diagonalizable (WJD) spatial model, which is used in FastMNMF2.

    You can specify the speech and noise models separately.
    The speech model is 'NMF', 'FreqInv', or 'DNN',
    and the noise model is 'TimeInv' or 'NMF'.
    For the DNN-based speech model, the initialization is important.
    Please use 'init_SCM = twostep', which first run FastBSS2 with FreqInv and TimeInv
    and then run FastBSS2 with DNN by using the estimated variables as initial values.

    For updating the spatial model, you can select the algorithm from 'IP' and 'ISS'.

    Meaning of the variables:
        F: n_freq = the number of frequency bins
        T: n_time = the number of time frames
        M: n_mic =  the number of microphones
        Ns: n_speech = the number of speakers
        Nn: n_noise = the number of noises
        N: n_speech + n_noise

        X_FTM: the observed multichannel complex spectrogram
        Q_FMM: diagonalizer that converts a spatial covariance matrix (SCM) to a diagonal matrix
        G_NM: diagonal elements of the diagonalized SCMs
        W_NsFKs: basis vectors for each source
        H_NsKsT: activations for each source
        PSD_NFT: power spectral densities of each source
        Y_FTM: sum_n PSD_NFT G_NM

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

    def __init__(
        self,
        n_speech=2,
        n_noise=0,
        speech_model=["NMF", "FreqInv", "DNN"][0],
        noise_model=["TimeInv", "NMF"][0],
        algo="ISS",
        xp=np,
        init_SCM="twostep",
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
        """initialize FastBSS2

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
        super(FastBSS2, self).__init__(xp=xp, n_bit=n_bit)
        self.n_speech = n_speech
        self.n_noise = n_noise
        self.n_source = self.n_speech + self.n_noise
        self.speech_model = speech_model
        self.noise_model = noise_model if self.n_noise > 0 else None
        self.algo = algo
        self.init_SCM = init_SCM

        self.n_basis = n_basis
        self.n_basis_noise = n_basis_noise

        self.speech_VAE = speech_VAE
        self.n_iter_z = n_iter_z
        self.lr = lr

        self.g_eps = g_eps
        self.n_iter_init = n_iter_init
        self.interval_norm = interval_norm

        self.method_name = "FastBSS2"

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
        )

        if self.n_bit == 32:
            modelname += "-bit=32"

        if hasattr(self, "file_id") and (self.file_id is not None):
            modelname += f"-ID={self.file_id}"

        return modelname

    def load_spectrogram(self, X_FTM, sample_rate=16000):
        super().load_spectrogram(X_FTM)
        if self.algo == "IP":
            self.XX_FTMM = self.xp.einsum("fti, ftj -> ftij", self.X_FTM, self.X_FTM.conj())

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
                self.W_noise_NnFKn = self.xp.random.rand(self.n_noise, self.n_freq, self.n_basis_noise).astype(
                    self.TYPE_FLOAT
                )
                self.H_noise_NnKnT = self.xp.random.rand(self.n_noise, self.n_basis_noise, self.n_time).astype(
                    self.TYPE_FLOAT
                )
            elif self.noise_model == "TimeInv":
                self.W_noise_NnFKn = self.xp.ones([self.n_noise, self.n_freq, 1]).astype(self.TYPE_FLOAT)
                self.H_noise_NnKnT = self.xp.ones([self.n_noise, 1, self.n_time]).astype(self.TYPE_FLOAT)
        self.PSD_NFT = self.xp.zeros([self.n_source, self.n_freq, self.n_time], dtype=self.TYPE_FLOAT)

    def init_spatial_model(self):
        self.start_idx = 0
        self.Q_FMM = self.xp.tile(self.xp.eye(self.n_mic), [self.n_freq, 1, 1]).astype(self.TYPE_COMPLEX)
        self.G_NM = self.xp.ones([self.n_source, self.n_mic], dtype=self.TYPE_FLOAT) * self.g_eps
        for m in range(self.n_mic):
            self.G_NM[m % self.n_source, m] = 1

        if "circular" in self.init_SCM:
            pass

        elif "obs" in self.init_SCM:
            if hasattr(self, "XX_FTMM"):
                XX_FMM = self.XX_FTMM.sum(axis=1)
            else:
                XX_FMM = self.xp.einsum("fti, ftj -> fij", self.X_FTM, self.X_FTM.conj())
            _, eig_vec_FMM = self.xp.linalg.eigh(XX_FMM)
            eig_vec_FMM = eig_vec_FMM[:, :, ::-1]
            self.Q_FMM = self.xp.asarray(eig_vec_FMM).transpose(0, 2, 1).conj()

            if self.speech_model == "DNN":
                if self.n_speech > 1:
                    raise ValueError("Error: init_SCM = obs is only for speech enhancement")
                power_speech_FT = self.xp.asarray(self.xp.abs(self.X_FTM[..., 0]) ** 2).astype(self.xp.float32)
                power_speech_FT /= power_speech_FT.sum(axis=0).mean(axis=0)
                with torch.set_grad_enabled(False):
                    self.Z_NsDT = (
                        self.speech_VAE.encode_(torch.as_tensor(power_speech_FT + EPS, device=self.torch_device))
                        .detach()
                        .unsqueeze(0)
                    )
                    self.Z_NsDT.requires_grad = True
                    self.z_optimizer = optim.AdamW([self.Z_NsDT], lr=self.lr)
                    self.power_speech_NsFT = self.xp.asarray(self.speech_VAE.decode_(self.Z_NsDT))

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

            separater_init = FastBSS2(
                n_speech=self.n_speech,
                n_noise=self.n_noise,
                speech_model="FreqInv",
                noise_model="TimeInv",
                init_SCM="circular",
                algo=self.algo,
                xp=self.xp,
                n_bit=self.n_bit,
                g_eps=self.g_eps,
            )
            separater_init.file_id = self.file_id
            separater_init.load_spectrogram(self.X_FTM)
            separater_init.solve(n_iter=self.start_idx, save_wav=False)
            self.Q_FMM = separater_init.Q_FMM
            self.G_NM = separater_init.G_NM

            if self.speech_model == "DNN":
                power_speech_NsFT = self.xp.asarray(
                    self.xp.abs(separater_init.separated_spec[: self.n_speech]) ** 2
                ).astype(self.xp.float32)
                power_speech_NsFT /= power_speech_NsFT.sum(axis=1).mean(axis=1)[:, None, None]
                with torch.set_grad_enabled(False):
                    self.Z_NsDT = self.speech_VAE.encode_(
                        torch.as_tensor(power_speech_NsFT + EPS, device=self.torch_device)
                    ).detach()
                    self.Z_NsDT.requires_grad = True
                    self.z_optimizer = optim.AdamW([self.Z_NsDT], lr=self.lr)
                    self.power_speech_NsFT = self.xp.asarray(self.speech_VAE.decode_(self.Z_NsDT))

        else:
            print(f"Please specify how to initialize covariance matrix {separater.init_SCM}")
            raise ValueError

        self.G_NM /= self.G_NM.sum(axis=1)[:, None]
        self.normalize()

    def calculate_Qx(self):
        self.Qx_FTM = self.xp.einsum("fmi, fti -> ftm", self.Q_FMM, self.X_FTM)
        self.Qx_power_FTM = self.xp.abs(self.Qx_FTM) ** 2

    def calculate_PSD(self):
        if self.speech_model == "NMF":
            self.PSD_NFT[: self.n_speech] = self.W_NsFKs @ self.H_NsKsT + EPS
        elif self.speech_model == "FreqInv":
            self.PSD_NFT[: self.n_speech] = self.PSD_NsT[:, None] + EPS
        elif self.speech_model == "DNN":
            self.PSD_NFT[: self.n_speech] = self.U_NsF[:, :, None] * self.V_NsT[:, None] * self.power_speech_NsFT + EPS

        if self.n_noise > 0:
            self.PSD_NFT[self.n_speech :] = self.W_noise_NnFKn @ self.H_noise_NnKnT + EPS

    def calculate_Y(self):
        self.Y_FTM = self.xp.einsum("nft, nm -> ftm", self.PSD_NFT, self.G_NM) + EPS

    def update(self):
        self.update_PSD()
        self.update_G()
        self.update_Q()
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
        tmp1_NsFT = self.xp.einsum("nm, ftm -> nft", self.G_NM[: self.n_speech], self.Qx_power_FTM / (self.Y_FTM**2))
        tmp2_NsFT = self.xp.einsum("nm, ftm -> nft", self.G_NM[: self.n_speech], 1 / self.Y_FTM)
        numerator = self.xp.einsum("nkt, nft -> nfk", self.H_NsKsT, tmp1_NsFT)
        denominator = self.xp.einsum("nkt, nft -> nfk", self.H_NsKsT, tmp2_NsFT)
        self.W_NsFKs *= self.xp.sqrt(numerator / denominator)
        self.calculate_PSD()
        self.calculate_Y()

        tmp1_NsFT = self.xp.einsum("nm, ftm -> nft", self.G_NM[: self.n_speech], self.Qx_power_FTM / (self.Y_FTM**2))
        tmp2_NsFT = self.xp.einsum("nm, ftm -> nft", self.G_NM[: self.n_speech], 1 / self.Y_FTM)
        numerator = self.xp.einsum("nfk, nft -> nkt", self.W_NsFKs, tmp1_NsFT)
        denominator = self.xp.einsum("nfk, nft -> nkt", self.W_NsFKs, tmp2_NsFT)
        self.H_NsKsT *= self.xp.sqrt(numerator / denominator)
        self.calculate_PSD()
        self.calculate_Y()

    def update_PSD_NMF_noise(self):
        tmp1_NnFT = self.xp.einsum("nm, ftm -> nft", self.G_NM[self.n_speech :], self.Qx_power_FTM / (self.Y_FTM**2))
        tmp2_NnFT = self.xp.einsum("nm, ftm -> nft", self.G_NM[self.n_speech :], 1 / self.Y_FTM)
        numerator = self.xp.einsum("nkt, nft -> nfk", self.H_noise_NnKnT, tmp1_NnFT)
        denominator = self.xp.einsum("nkt, nft -> nfk", self.H_noise_NnKnT, tmp2_NnFT)
        self.W_noise_NnFKn *= self.xp.sqrt(numerator / denominator)
        self.calculate_PSD()
        self.calculate_Y()

        tmp1_NnFT = self.xp.einsum("nm, ftm -> nft", self.G_NM[self.n_speech :], self.Qx_power_FTM / (self.Y_FTM**2))
        tmp2_NnFT = self.xp.einsum("nm, ftm -> nft", self.G_NM[self.n_speech :], 1 / self.Y_FTM)
        numerator = self.xp.einsum("nfk, nft -> nkt", self.W_noise_NnFKn, tmp1_NnFT)
        denominator = self.xp.einsum("nfk, nft -> nkt", self.W_noise_NnFKn, tmp2_NnFT)
        self.H_noise_NnKnT *= self.xp.sqrt(numerator / denominator)
        self.calculate_PSD()
        self.calculate_Y()

    def update_PSD_FreqInv(self):
        GXY2_NsT = self.xp.einsum(
            "nm, tm -> nt", self.G_NM[: self.n_speech], (self.Qx_power_FTM / (self.Y_FTM**2)).sum(axis=0)
        )
        GY_NsT = self.xp.einsum("nm, tm -> nt", self.G_NM[: self.n_speech], (1 / self.Y_FTM).sum(axis=0))
        self.PSD_NsT *= self.xp.sqrt(GXY2_NsT / GY_NsT)
        self.calculate_PSD()
        self.calculate_Y()

    def update_PSD_DNN(self):
        def loss_fn(Y_noise_FTM_torch, G_NM_torch, UV_NsFT_torch):  # for update Z by backprop
            power_speech_NsFT = self.speech_VAE.decode_(self.Z_NsDT)
            PSD_tmp_NsFT = UV_NsFT_torch * power_speech_NsFT  # + EPS

            Y_tmp_FTM = (
                (PSD_tmp_NsFT[..., None] * G_NM_torch[: self.n_speech, None, None]).sum(axis=0)
                + Y_noise_FTM_torch
                + EPS
            )
            return (
                torch.log(Y_tmp_FTM) + torch.as_tensor(self.Qx_power_FTM, device=self.torch_device) / Y_tmp_FTM
            ).sum() / (self.n_freq * self.n_mic)

        if self.n_noise > 0:
            Y_noise_FTM_torch = (
                self.PSD_NFT[self.n_speech :, :, :, None] * self.G_NM[self.n_speech :, None, None]
            ).sum(axis=0)
        else:
            Y_noise_FTM_torch = self.xp.zeros_like(self.X_FTM, dtype=self.TYPE_FLOAT)
        Y_noise_FTM_torch = torch.as_tensor(Y_noise_FTM_torch, device=self.torch_device)
        G_NM_torch = torch.as_tensor(self.G_NM, device=self.torch_device)
        UV_NsFT_torch = torch.as_tensor(self.U_NsF[:, :, None] * self.V_NsT[:, None], device=self.torch_device)

        for it in range(self.n_iter_z):
            self.z_optimizer.zero_grad()
            loss = loss_fn(Y_noise_FTM_torch, G_NM_torch, UV_NsFT_torch)
            loss.backward()
            self.z_optimizer.step()

        with torch.set_grad_enabled(False):
            self.power_speech_NsFT = self.xp.asarray(self.speech_VAE.decode_(self.Z_NsDT))
        self.calculate_PSD()
        self.calculate_Y()

        XY2_FTM = self.Qx_power_FTM / (self.Y_FTM**2)
        VZG_NsFTM = (self.V_NsT[..., None] * self.G_NM[: self.n_speech, None])[:, None] * self.power_speech_NsFT[
            ..., None
        ]
        a_U_NsF = (VZG_NsFTM * XY2_FTM[None]).sum(axis=(2, 3))
        b_U_NsF = (VZG_NsFTM / self.Y_FTM[None]).sum(axis=(2, 3))

        self.U_NsF *= self.xp.sqrt(a_U_NsF / b_U_NsF)
        self.calculate_PSD()
        self.calculate_Y()

        XY2_FTM = self.Qx_power_FTM / (self.Y_FTM**2)
        GXY2_NFT = (self.G_NM[: self.n_speech, None, None] * XY2_FTM[None]).sum(axis=-1)
        GYinv_NFT = (self.G_NM[: self.n_speech, None, None] / self.Y_FTM[None]).sum(axis=-1)
        a_V_NsT = ((self.U_NsF[:, :, None] * self.power_speech_NsFT) * GXY2_NFT).sum(axis=1)
        b_V_NsT = ((self.U_NsF[:, :, None] * self.power_speech_NsFT) * GYinv_NFT).sum(axis=1)

        self.V_NsT *= self.xp.sqrt(a_V_NsT / b_V_NsT)
        self.calculate_PSD()
        self.calculate_Y()

    def update_G(self):
        numerator = self.xp.einsum("nft, ftm -> nm", self.PSD_NFT, self.Qx_power_FTM / (self.Y_FTM**2))
        denominator = self.xp.einsum("nft, ftm -> nm", self.PSD_NFT, 1 / self.Y_FTM)
        self.G_NM *= self.xp.sqrt(numerator / denominator)
        self.calculate_Y()

    def update_Q(self):
        if self.algo == "IP":
            for m in range(self.n_mic):
                V_FMM = self.xp.einsum("ftij, ft -> fij", self.XX_FTMM, 1 / self.Y_FTM[..., m]) / self.n_time
                tmp_FM = self.xp.linalg.inv(self.Q_FMM @ V_FMM)[..., m]
                self.Q_FMM[:, m] = (
                    tmp_FM / self.xp.sqrt(self.xp.einsum("fi, fij, fj -> f", tmp_FM.conj(), V_FMM, tmp_FM))[:, None]
                ).conj()
        elif self.algo == "ISS":
            for m in range(self.n_mic):
                QxQx_FTM = self.Qx_FTM * self.Qx_FTM[:, :, m, None].conj()
                V_tmp_FxM = (QxQx_FTM[:, :, m, None] / self.Y_FTM).mean(axis=1)
                V_FxM = (QxQx_FTM / self.Y_FTM).mean(axis=1) / V_tmp_FxM
                V_FxM[:, m] = 1 - 1 / self.xp.sqrt(V_tmp_FxM[:, m])
                self.Qx_FTM -= self.xp.einsum("fm, ft -> ftm", V_FxM, self.Qx_FTM[:, :, m])
                self.Q_FMM -= self.xp.einsum("fi, fj -> fij", V_FxM, self.Q_FMM[:, m])
        else:
            raise ValueError("algo has to be IP or ISS")
        self.calculate_Qx()

    def normalize(self):
        if self.speech_model in ["NMF", "DNN"]:
            phi_F = self.xp.einsum("fij, fij -> f", self.Q_FMM, self.Q_FMM.conj()).real / self.n_mic
            self.Q_FMM /= self.xp.sqrt(phi_F)[:, None, None]
            if self.speech_model == "NMF":
                self.W_NsFKs /= phi_F[None, :, None]
            elif self.speech_model == "DNN":
                self.U_NsF /= phi_F[None]
            if self.n_noise > 0:
                self.W_noise_NnFKn /= phi_F[None, :, None]

        mu_N = self.G_NM.sum(axis=1)
        self.G_NM /= mu_N[:, None]
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

        self.calculate_Qx()
        self.calculate_PSD()
        self.calculate_Y()

    def separate(self, mic_index=MIC_INDEX):
        Y_NsFTM = self.xp.einsum("nft, nm -> nftm", self.PSD_NFT[: self.n_speech], self.G_NM[: self.n_speech])
        self.Y_FTM = Y_NsFTM.sum(axis=0)
        self.Qx_FTM = self.xp.einsum("fmi, fti -> ftm", self.Q_FMM, self.X_FTM)
        Qinv_FMM = self.xp.linalg.inv(self.Q_FMM)

        self.separated_spec = self.xp.einsum(
            "fj, ftj, nftj -> nft", Qinv_FMM[:, mic_index], self.Qx_FTM / self.Y_FTM, Y_NsFTM
        )
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
            self.n_noise, _, self.n_basis_noise = self.W_noise_NnFKn.shape
            if self.n_basis_noise == 1:
                self.noise_model = "TimeInv"
            else:
                self.noise_model = "NMF"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_fname", type=str, help="filename of the multichannel observed signals")

    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--n_speech", type=int, default=3, help="number of speech")
    parser.add_argument("--n_noise", type=int, default=0, help="number of noise")
    parser.add_argument("--speech_model", type=str, default="DNN", help="NMF, FreqInv, DNN")
    parser.add_argument("--noise_model", type=str, default="NMF", help="TimeInv, NMF")
    parser.add_argument("--n_iter", type=int, default=100, help="number of iteration")
    parser.add_argument("--init_SCM", type=str, default="twostep", help="circular or twostep")
    parser.add_argument("--algo", type=str, default="IP", help="IP or ISS")

    # Only for NMF source model
    parser.add_argument("--n_basis", type=int, default=16, help="number of basis for speech")
    parser.add_argument("--n_basis_noise", type=int, default=16, help="number of basis for noise")

    # Only for DNN source model
    parser.add_argument("--n_iter_z", type=int, default=5, help="number of iteration for updating Z")

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
        state_dict = torch.load(nn_fname, map_location=device)
        speech_VAE.load_state_dict(state_dict["net_state_dict"])
        speech_VAE.to(device)

        # The input length must be a multiple of 4
        spec_FTM = spec_FTM[:, : spec_FTM.shape[1] // 4 * 4]
    else:
        speech_VAE = None

    # Start separation
    separater = FastBSS2(speech_VAE=speech_VAE, xp=xp, **vars(args))
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
