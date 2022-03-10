#! /usr/bin/env python3
# coding: utf-8

import sys
import os
import numpy as np
from pathlib import Path

sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))
from Base import EPS, MIC_INDEX, Base, MultiSTFT


class FastMNMF1(Base):
    """
    The blind souce separation using FastMNMF1

    X_FTM: the observed complex spectrogram
    Q_FMM: diagonalizer that converts SCMs to diagonal matrices
    G_NFM: diagonal elements of the diagonalized SCMs
    W_NFK: basis vectors
    H_NKT: activations
    PSD_NFT: power spectral densities
    Qx_power_FTM: power spectra of Q_FMM times X_FTM
    Y_FTM: sum of (PSD_NFT x G_NFM) over all sources
    """

    def __init__(
        self,
        n_source,
        n_basis=8,
        init_SCM="twostep",
        algo="IP",
        n_iter_init=10,
        g_eps=5e-2,
        interval_norm=10,
        n_bit=64,
        xp=np,
        seed=0,
    ):
        """Initialize FastMNMF1

        Parameters:
        -----------
            n_source: int
                The number of sources.
            n_basis: int
                The number of bases for the NMF-based source model.
            init_SCM: str ('circular', 'obs', 'twostep')
                How to initialize SCM.
                'obs' is for the case that one speech is dominant in the mixture.
            algo: str (IP, ISS)
                How to update Q.
            n_iter_init: int
                The number of iteration for the first step in twostep initialization.
            xp : numpy or cupy
        """
        super().__init__(xp=xp, n_bit=n_bit, seed=seed)
        self.n_source = n_source
        self.n_basis = n_basis
        self.init_SCM = init_SCM
        self.g_eps = g_eps
        self.algo = algo
        self.interval_norm = interval_norm
        self.n_iter_init = n_iter_init
        self.save_param_list += ["W_NFK", "H_NKT", "G_NFM", "Q_FMM"]

        if self.algo == "IP":
            self.method_name = "FastMNMF1_IP"
        elif "ISS" in algo:
            self.method_name = "FastMNMF1_ISS"
        else:
            raise ValueError("algo must be IP or ISS")

    def __str__(self):
        init = f"twostep_{self.n_iter_init}it" if self.init_SCM == "twostep" else self.init_SCM
        filename_suffix = (
            f"M={self.n_mic}-S={self.n_source}-F={self.n_freq}-K={self.n_basis}"
            f"-init={init}-g={self.g_eps}-bit={self.n_bit}-intv_norm={self.interval_norm}"
        )
        if hasattr(self, "file_id"):
            filename_suffix += f"-ID={self.file_id}"
        return filename_suffix

    def load_spectrogram(self, X_FTM, sample_rate=16000):
        super().load_spectrogram(X_FTM, sample_rate=sample_rate)
        if self.algo == "IP":
            self.XX_FTMM = self.xp.einsum("fti, ftj -> ftij", X_FTM, X_FTM.conj())

    def init_source_model(self):
        self.W_NFK = self.xp.random.rand(self.n_source, self.n_freq, self.n_basis).astype(self.TYPE_FLOAT)
        self.H_NKT = self.xp.random.rand(self.n_source, self.n_basis, self.n_time).astype(self.TYPE_FLOAT)

    def init_spatial_model(self):
        self.start_idx = 0
        self.Q_FMM = self.xp.tile(self.xp.eye(self.n_mic), [self.n_freq, 1, 1]).astype(self.TYPE_COMPLEX)
        self.G_NFM = self.xp.maximum(
            self.g_eps, self.xp.zeros([self.n_source, self.n_freq, self.n_mic], dtype=self.TYPE_FLOAT)
        )
        for m in range(self.n_mic):
            self.G_NFM[m % self.n_source, :, m] = 1

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
        elif "twostep" == self.init_SCM:
            if self.n_iter_init >= self.n_iter:
                print(
                    "\n------------------------------------------------------------------\n"
                    f"Warning: n_iter_init must be smaller than n_iter (= {self.n_iter}).\n"
                    f"n_iter_init is changed from {self.n_iter_init} to {self.n_iter // 3}"
                    "\n------------------------------------------------------------------\n"
                )
                self.n_iter_init = self.n_iter // 3

            self.start_idx = self.n_iter_init

            separater_init = FastMNMF1(
                n_source=self.n_source,
                n_basis=2,
                init_SCM="circular",
                xp=self.xp,
                n_bit=self.n_bit,
                g_eps=self.g_eps,
            )
            separater_init.load_spectrogram(self.X_FTM)
            separater_init.solve(n_iter=self.start_idx, save_wav=False)

            self.Q_FMM = separater_init.Q_FMM
            self.G_NFM = separater_init.G_NFM
        else:
            raise ValueError("init_SCM should be circular, obs, or twostep.")

        self.G_NFM /= self.G_NFM.sum(axis=2)[..., None]
        self.normalize()

    def calculate_Qx(self):
        self.Qx_FTM = self.xp.einsum("fmi, fti -> ftm", self.Q_FMM, self.X_FTM)
        self.Qx_power_FTM = self.xp.abs(self.Qx_FTM) ** 2

    def calculate_PSD(self):
        self.PSD_NFT = self.W_NFK @ self.H_NKT + EPS

    def calculate_Y(self):
        self.Y_FTM = self.xp.einsum("nft, nfm -> ftm", self.PSD_NFT, self.G_NFM) + EPS

    def update(self):
        self.update_WH()
        self.update_G()
        if self.algo == "IP":
            self.update_Q_IP()
        else:
            self.update_Q_ISS()
        if self.it % self.interval_norm == 0:
            self.normalize()
        else:
            self.calculate_Qx()

    def update_WH(self):
        tmp1_NFT = self.xp.einsum("nfm, ftm -> nft", self.G_NFM, self.Qx_power_FTM / (self.Y_FTM**2))
        tmp2_NFT = self.xp.einsum("nfm, ftm -> nft", self.G_NFM, 1 / self.Y_FTM)
        numerator = self.xp.einsum("nkt, nft -> nfk", self.H_NKT, tmp1_NFT)
        denominator = self.xp.einsum("nkt, nft -> nfk", self.H_NKT, tmp2_NFT)
        self.W_NFK *= self.xp.sqrt(numerator / denominator)
        self.calculate_PSD()
        self.calculate_Y()

        tmp1_NFT = self.xp.einsum("nfm, ftm -> nft", self.G_NFM, self.Qx_power_FTM / (self.Y_FTM**2))
        tmp2_NFT = self.xp.einsum("nfm, ftm -> nft", self.G_NFM, 1 / self.Y_FTM)
        numerator = self.xp.einsum("nfk, nft -> nkt", self.W_NFK, tmp1_NFT)
        denominator = self.xp.einsum("nfk, nft -> nkt", self.W_NFK, tmp2_NFT)
        self.H_NKT *= self.xp.sqrt(numerator / denominator)
        self.calculate_PSD()
        self.calculate_Y()

    def update_G(self):
        numerator = self.xp.einsum("nft, ftm -> nfm", self.PSD_NFT, self.Qx_power_FTM / (self.Y_FTM**2))
        denominator = self.xp.einsum("nft, ftm -> nfm", self.PSD_NFT, 1 / self.Y_FTM)
        self.G_NFM *= self.xp.sqrt(numerator / denominator)
        self.calculate_Y()

    def update_Q_IP(self):
        for m in range(self.n_mic):
            V_FMM = self.xp.einsum("ftij, ft -> fij", self.XX_FTMM, 1 / self.Y_FTM[..., m]) / self.n_time
            tmp_FM = self.xp.linalg.inv(self.Q_FMM @ V_FMM)[..., m]
            self.Q_FMM[:, m] = (
                tmp_FM / self.xp.sqrt(self.xp.einsum("fi, fij, fj -> f", tmp_FM.conj(), V_FMM, tmp_FM))[:, None]
            ).conj()

    def update_Q_ISS(self):
        for m in range(self.n_mic):
            QxQx_FTM = self.Qx_FTM * self.Qx_FTM[:, :, m, None].conj()
            V_tmp_FxM = (QxQx_FTM[:, :, m, None] / self.Y_FTM).mean(axis=1)
            V_FxM = (QxQx_FTM / self.Y_FTM).mean(axis=1) / V_tmp_FxM
            V_FxM[:, m] = 1 - 1 / self.xp.sqrt(V_tmp_FxM[:, m])
            self.Qx_FTM -= self.xp.einsum("fm, ft -> ftm", V_FxM, self.Qx_FTM[:, :, m])
            self.Q_FMM -= self.xp.einsum("fi, fj -> fij", V_FxM, self.Q_FMM[:, m])

    def normalize(self):
        phi_F = self.xp.einsum("fij, fij -> f", self.Q_FMM, self.Q_FMM.conj()).real / self.n_mic
        self.Q_FMM /= self.xp.sqrt(phi_F)[:, None, None]
        self.G_NFM /= phi_F[None, :, None]

        mu_NF = self.G_NFM.sum(axis=2)
        self.G_NFM /= mu_NF[..., None]
        self.W_NFK *= mu_NF[..., None]

        nu_NK = self.W_NFK.sum(axis=1)
        self.W_NFK /= nu_NK[:, None]
        self.H_NKT *= nu_NK[:, :, None]

        self.calculate_Qx()
        self.calculate_PSD()
        self.calculate_Y()

    def separate(self, mic_index=MIC_INDEX):
        Y_NFTM = self.xp.einsum("nft, nfm -> nftm", self.PSD_NFT, self.G_NFM)
        self.Y_FTM = Y_NFTM.sum(axis=0)
        self.Qx_FTM = self.xp.einsum("fmi, fti -> ftm", self.Q_FMM, self.X_FTM)
        Qinv_FMM = self.xp.linalg.inv(self.Q_FMM)

        self.separated_spec = self.xp.einsum(
            "fj, ftj, nftj -> nft", Qinv_FMM[:, mic_index], self.Qx_FTM / self.Y_FTM, Y_NFTM
        )
        return self.separated_spec

    def calculate_log_likelihood(self):
        log_likelihood = (
            -(self.Qx_power_FTM / self.Y_FTM + self.xp.log(self.Y_FTM)).sum()
            + self.n_time * (self.xp.log(self.xp.linalg.det(self.Q_FMM @ self.Q_FMM.transpose(0, 2, 1).conj()))).sum()
        ).real
        return log_likelihood

    def load_param(self, filename):
        super().load_param(filename)

        self.n_source, self.n_freq, self.n_basis = self.W_NFK.shape
        _, _, self.n_time = self.H_NKT


if __name__ == "__main__":
    import soundfile as sf
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_fname", type=str, help="filename of the multichannel observed signals")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--n_fft", type=int, default=1024, help="number of frequencies")
    parser.add_argument("--n_source", type=int, default=3, help="number of noise")
    parser.add_argument("--n_basis", type=int, default=4, help="number of basis")
    parser.add_argument("--n_iter_init", type=int, default=30, help="nujmber of iteration used in twostep init")
    parser.add_argument(
        "--init_SCM",
        type=str,
        default="twostep",
        help="circular, obs (only for enhancement), twostep",
    )
    parser.add_argument("--n_iter", type=int, default=100, help="number of iteration")
    parser.add_argument("--n_mic", type=int, default=8, help="number of microphone")
    parser.add_argument("--n_bit", type=int, default=64, help="number of microphone")
    parser.add_argument("--algo", type=str, default="IP", help="the method for updating Q")
    args = parser.parse_args()

    if args.gpu < 0:
        import numpy as xp
    else:
        try:
            import cupy as xp

            print("Use GPU " + str(args.gpu))
            xp.cuda.Device(args.gpu).use()
        except ImportError:
            print("Warning: cupy is not installed. 'gpu' argument should be set to -1. Switched to CPU.\n")
            import numpy as xp

    separater = FastMNMF1(
        n_source=args.n_source,
        n_basis=args.n_basis,
        xp=xp,
        init_SCM=args.init_SCM,
        n_bit=args.n_bit,
        algo=args.algo,
        n_iter_init=args.n_iter_init,
    )

    wav, sample_rate = sf.read(args.input_fname)
    wav /= np.abs(wav).max() * 1.2
    M = min(len(wav), args.n_mic)
    spec_FTM = MultiSTFT(wav[:, :M], n_fft=args.n_fft)

    separater.file_id = args.input_fname.split("/")[-1].split(".")[0]
    separater.load_spectrogram(spec_FTM, sample_rate)
    separater.solve(
        n_iter=args.n_iter,
        save_dir="./",
        save_likelihood=False,
        save_param=False,
        save_wav=True,
        interval_save=5,
    )
