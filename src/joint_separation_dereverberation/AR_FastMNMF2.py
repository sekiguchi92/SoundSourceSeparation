#! /usr/bin/env python3
# coding: utf-8

import sys
import os
import numpy as np
from pathlib import Path

sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))
from Base import EPS, MIC_INDEX, Base, MultiSTFT


class AR_FastMNMF2(Base):
    """
    The joint blind souce separation and dereverberation method
    that integrates FastMNMF2 with the AR reverberation model.

    X_FTM: the observed complex spectrogram
    Q_FMM: diagonalizer that converts SCMs to diagonal matrices
    P_FMM: the matrix obtained by multiplying diagonalizer and AR filter
    G_NM: diagonal elements of the diagonalized SCMs
    W_NFK: basis vectors
    H_NKT: activations
    PSD_NFT: power spectral densities
    Px_power_FTM: power spectra of P_FMM times X_FTM
    Y_FTM: sum of (PSD_NFT x G_NM) over all sources
    """

    def __init__(
        self,
        n_source,
        n_basis=8,
        SCM="twostep",
        algo="IP",
        n_tap_AR=3,
        n_delay_AR=3,
        n_iter_init=10,
        n_bit=64,
        xp=np,
        g_eps=5e-2,
        interval_norm=10,
    ):
        """ Initialize AR_FastMNMF2

        Parameters:
        -----------
            n_source: int
                The number of sources.
            n_basis: int
                The number of bases for the NMF-based source model.
            SCM: str ('circular', 'obs', 'twostep')
                How to initialize SCM.
                'obs' is for the case that one speech is dominant in the mixture.
            algo: str (IP, ISS, ISS_Joint)
                How to update P (diagonalizer and AR filter).
            n_tap_AR: int
                Tap length for the AR model.
            n_delay_AR: int
                The index to indicate the beginning of the late reverberation.
            n_iter_init: int
                The number of iteration for the first step in twostep initialization.
        """
        super().__init__(xp=xp, n_bit=n_bit)
        self.n_source = n_source
        self.n_basis = n_basis
        self.SCM = SCM
        self.n_tap_AR = n_tap_AR
        self.n_delay_AR = n_delay_AR
        self.g_eps = g_eps
        self.algo = algo
        self.interval_norm = interval_norm
        self.n_iter_init = n_iter_init
        self.save_param_list += ["W_NFK", "H_NKT", "G_NM", "P_FxMxMLa"]

        if self.algo == "IP":
            self.method_name = "AR_FastMNMF2_IP"
        elif (algo == "ISS_Joint") and (self.n_tap_AR > 0):
            self.method_name = "AR_FastMNMF2_ISS_Joint"
        elif "ISS" in algo:
            self.method_name = "AR_FastMNMF2_ISS"
            self.algo = "ISS"
        else:
            raise ValueError("algo must be IP, ISS_Joint, or ISS")

    def __str__(self):
        init = f"twostep_{self.n_iter_init}it" if self.SCM == "twostep" else self.SCM
        filename_suffix = (
            f"M={self.n_mic}-S={self.n_source}-F={self.n_freq}-K={self.n_basis}"
            f"-init={init}-Dar={self.n_delay_AR}-Lar={self.n_tap_AR}"
            f"-g={self.g_eps}-bit={self.n_bit}-intv_norm={self.interval_norm}"
        )
        if hasattr(self, "file_id"):
            filename_suffix += f"-ID={self.file_id}"
        return filename_suffix

    def load_spectrogram(self, X_FTM, sample_rate=16000):
        super(AR_FastMNMF2, self).load_spectrogram(X_FTM=X_FTM, sample_rate=sample_rate)

        self.Xbar_FxTxMLa = self.xp.zeros(
            [self.n_freq, self.n_time, self.n_mic * (self.n_tap_AR + 1)], dtype=self.TYPE_COMPLEX
        )
        self.Xbar_FxTxMLa[:, :, : self.n_mic] = self.X_FTM
        for i in range(self.n_tap_AR):
            self.Xbar_FxTxMLa[:, self.n_delay_AR + i :, (i + 1) * self.n_mic : (i + 2) * self.n_mic] = self.X_FTM[
                :, : -(self.n_delay_AR + i)
            ]

    def init_PSD(self):
        self.W_NFK = self.xp.random.rand(self.n_source, self.n_freq, self.n_basis).astype(self.TYPE_FLOAT)
        self.H_NKT = self.xp.random.rand(self.n_source, self.n_basis, self.n_time).astype(self.TYPE_FLOAT)

    def init_SCM(self):
        self.start_idx = 0
        self.Q_FMM = self.xp.tile(self.xp.eye(self.n_mic), [self.n_freq, 1, 1]).astype(self.TYPE_COMPLEX)
        self.P_FxMxMLa = self.xp.zeros(
            [self.n_freq, self.n_mic, self.n_mic * (self.n_tap_AR + 1)], dtype=self.TYPE_COMPLEX
        )
        self.P_FxMxMLa[:, :, : self.n_mic] = self.Q_FMM
        self.G_NM = self.xp.maximum(self.g_eps, self.xp.zeros([self.n_source, self.n_mic], dtype=self.TYPE_FLOAT))
        for m in range(self.n_mic):
            self.G_NM[m % self.n_source, m] = 1

        if "circular" in self.SCM:
            pass
        elif "obs" in self.SCM:
            XX_FMM = self.xp.einsum("fti, ftj -> fij", self.X_FTM, self.X_FTM.conj())
            _, eig_vec_FMM = self.xp.linalg.eigh(XX_FMM)
            eig_vec_FMM = eig_vec_FMM[:, :, ::-1]
            self.Q_FMM = self.xp.asarray(eig_vec_FMM).transpose(0, 2, 1).conj()
            self.P_FxMxMLa[:, :, : self.n_mic] = self.Q_FMM
        elif "twostep" == self.SCM:
            if self.n_iter_init >= self.n_iter:
                print(
                    "\n------------------------------------------------------------------\n"
                    f"Warning: n_iter_init must be smaller than n_iter (= {self.n_iter}).\n"
                    f"n_iter_init is changed from {self.n_iter_init} to {self.n_iter // 3}"
                    "\n------------------------------------------------------------------\n"
                )
                self.n_iter_init = self.n_iter // 3

            self.start_idx = self.n_iter_init
            SCM_for_init = "circular"

            separater_init = AR_FastMNMF2(
                n_source=self.n_source,
                n_basis=2,
                SCM=SCM_for_init,
                xp=self.xp,
                n_bit=self.n_bit,
                n_tap_AR=self.n_tap_AR,
                n_delay_AR=self.n_delay_AR,
                g_eps=self.g_eps,
            )
            separater_init.load_spectrogram(self.X_FTM)
            separater_init.solve(n_iter=self.start_idx, save_wav=False)

            self.P_FxMxMLa = separater_init.P_FxMxMLa
            self.Q_FMM = self.P_FxMxMLa[..., : self.n_mic]
            self.G_NM = separater_init.G_NM
        else:
            raise ValueError("SCM should be circular, obs, or twostep.")

        self.G_NM /= self.G_NM.sum(axis=1)[:, None]
        self.normalize()

    def calculate_Px(self):
        self.Px_FTM = self.xp.einsum("fmi, fti -> ftm", self.P_FxMxMLa, self.Xbar_FxTxMLa)
        self.Px_power_FTM = self.xp.abs(self.Px_FTM) ** 2

    def calculate_PSD(self):
        self.PSD_NFT = self.W_NFK @ self.H_NKT + EPS

    def calculate_Y(self):
        self.Y_FTM = self.xp.einsum("nft, nm -> ftm", self.PSD_NFT, self.G_NM) + EPS

    def update(self):
        self.update_WH()
        self.update_G()
        if self.algo == "IP":
            self.update_P_IP()
        else:
            self.update_P_ISS()
        if self.it % self.interval_norm == 0:
            self.normalize()
        else:
            self.calculate_Px()

    def update_WH(self):
        tmp1_NFT = self.xp.einsum("nm, ftm -> nft", self.G_NM, self.Px_power_FTM / (self.Y_FTM ** 2))
        tmp2_NFT = self.xp.einsum("nm, ftm -> nft", self.G_NM, 1 / self.Y_FTM)
        numerator = self.xp.einsum("nkt, nft -> nfk", self.H_NKT, tmp1_NFT)
        denominator = self.xp.einsum("nkt, nft -> nfk", self.H_NKT, tmp2_NFT)
        self.W_NFK *= self.xp.sqrt(numerator / denominator)
        self.calculate_PSD()
        self.calculate_Y()

        tmp1_NFT = self.xp.einsum("nm, ftm -> nft", self.G_NM, self.Px_power_FTM / (self.Y_FTM ** 2))
        tmp2_NFT = self.xp.einsum("nm, ftm -> nft", self.G_NM, 1 / self.Y_FTM)
        numerator = self.xp.einsum("nfk, nft -> nkt", self.W_NFK, tmp1_NFT)
        denominator = self.xp.einsum("nfk, nft -> nkt", self.W_NFK, tmp2_NFT)
        self.H_NKT *= self.xp.sqrt(numerator / denominator)
        self.calculate_PSD()
        self.calculate_Y()

    def update_G(self):
        numerator = self.xp.einsum("nft, ftm -> nm", self.PSD_NFT, self.Px_power_FTM / (self.Y_FTM ** 2))
        denominator = self.xp.einsum("nft, ftm -> nm", self.PSD_NFT, 1 / self.Y_FTM)
        self.G_NM *= self.xp.sqrt(numerator / denominator)
        self.calculate_Y()

    def update_P_IP(self):
        for m in range(self.n_mic):
            Vinv_FxMLtxMLt = self.xp.linalg.inv(
                self.xp.einsum(
                    "fti, ftj, ft -> fij", self.Xbar_FxTxMLa, self.Xbar_FxTxMLa.conj(), 1 / self.Y_FTM[..., m]
                )
                / self.n_time
            )
            u_FM = self.xp.linalg.inv(self.P_FxMxMLa[:, :, : self.n_mic])[:, :, m]
            denominator = self.xp.sqrt(
                self.xp.einsum("fi, fij, fj -> f", u_FM.conj(), Vinv_FxMLtxMLt[:, : self.n_mic, : self.n_mic], u_FM)
            )
            self.P_FxMxMLa[:, m] = self.xp.einsum(
                "fi, f -> fi",
                self.xp.einsum("fij, fj -> fi", Vinv_FxMLtxMLt[..., : self.n_mic], u_FM),
                1 / denominator,
            ).conj()

        self.Q_FMM = self.P_FxMxMLa[:, :, : self.n_mic]

    def update_P_ISS(self):
        for m in range(self.n_mic):
            PxPx_FTM = self.Px_FTM * self.Px_FTM[:, :, m, None].conj()
            V_tmp_FxM = (PxPx_FTM[:, :, m, None] / self.Y_FTM).mean(axis=1)
            V_FxM = (PxPx_FTM / self.Y_FTM).mean(axis=1) / V_tmp_FxM
            V_FxM[:, m] = 1 - 1 / self.xp.sqrt(V_tmp_FxM[:, m])
            self.Px_FTM -= self.xp.einsum("fm, ft -> ftm", V_FxM, self.Px_FTM[:, :, m])
            self.P_FxMxMLa -= self.xp.einsum("fi, fj -> fij", V_FxM, self.P_FxMxMLa[:, m])

        if self.n_tap_AR > 0:
            if self.algo == "ISS_Joint":
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

            elif self.algo == "ISS":
                for m in range(self.n_mic, (self.n_tap_AR + 1) * self.n_mic):
                    a_FxM = ((self.Px_FTM / self.Y_FTM) * self.Xbar_FxTxMLa[:, :, m, None].conj()).sum(axis=1)
                    b_FxM = ((self.xp.abs(self.Xbar_FxTxMLa[:, :, m]) ** 2)[:, :, None] / self.Y_FTM).sum(axis=1)
                    V_FxM = a_FxM / b_FxM
                    self.P_FxMxMLa[:, :, m] -= V_FxM
                    self.Px_FTM -= V_FxM[:, None] * self.Xbar_FxTxMLa[:, :, m, None]

        self.Q_FMM = self.P_FxMxMLa[:, :, : self.n_mic]

    def normalize(self):
        phi_F = self.xp.einsum("fij, fij -> f", self.Q_FMM, self.Q_FMM.conj()).real / self.n_mic
        self.P_FxMxMLa /= self.xp.sqrt(phi_F)[:, None, None]
        self.W_NFK /= phi_F[None, :, None]

        mu_N = self.G_NM.sum(axis=1)
        self.G_NM /= mu_N[:, None]
        self.W_NFK *= mu_N[:, None, None]

        nu_NK = self.W_NFK.sum(axis=1)
        self.W_NFK /= nu_NK[:, None]
        self.H_NKT *= nu_NK[:, :, None]

        self.calculate_Px()
        self.calculate_PSD()
        self.calculate_Y()

    def separate(self, mic_index=MIC_INDEX):
        Y_NFTM = self.xp.einsum("nft, nm -> nftm", self.PSD_NFT, self.G_NM)
        self.Y_FTM = Y_NFTM.sum(axis=0)
        self.Px_FTM = self.xp.einsum("fmi, fti -> ftm", self.P_FxMxMLa, self.Xbar_FxTxMLa)
        Qinv_FMM = self.xp.linalg.inv(self.Q_FMM)

        self.separated_spec = self.xp.einsum(
            "fj, ftj, nftj -> nft", Qinv_FMM[:, mic_index], self.Px_FTM / self.Y_FTM, Y_NFTM
        )
        return self.separated_spec

    def calculate_log_likelihood(self):
        log_likelihood = (
            -(self.Px_power_FTM / self.Y_FTM + self.xp.log(self.Y_FTM)).sum()
            + self.n_time * (self.xp.log(self.xp.linalg.det(self.Q_FMM @ self.Q_FMM.transpose(0, 2, 1).conj()))).sum()
        ).real
        return log_likelihood

    def load_param(self, filename):
        super(AR_FastMNMF2, self).load_param(filename)

        self.n_source, self.n_freq, self.n_basis = self.W_NFK.shape
        _, _, self.n_time = self.H_NKT
        self.n_tap_AR = (self.P_FxMxMLa.shape[2] / self.n_mic) - 1


if __name__ == "__main__":
    import argparse

    import soundfile as sf

    parser = argparse.ArgumentParser()
    parser.add_argument("input_fname", type=str, help="filename of the multichannel observed signals")
    parser.add_argument("--file_id", type=str, default="None", help="file id")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU ID. If -1, CPU is used.")
    parser.add_argument("--n_fft", type=int, default=1024, help="number of frequencies")
    parser.add_argument("--n_source", type=int, default=3, help="number of noise")
    parser.add_argument("--n_basis", type=int, default=4, help="number of basis")
    parser.add_argument("--n_tap_AR", type=int, default=4, help="number of basis of NMF")
    parser.add_argument("--n_delay_AR", type=int, default=3, help="number of basis of NMF")
    parser.add_argument("--SCM", type=str, default="twostep", help="circular, obs, twostep")
    parser.add_argument("--n_iter_init", type=int, default=30, help="nujmber of iteration used in twostep init")
    parser.add_argument("--n_iter", type=int, default=100, help="number of iteration")
    parser.add_argument("--n_mic", type=int, default=8, help="number of microphone")
    parser.add_argument("--n_bit", type=int, default=64, help="number of microphone")
    parser.add_argument("--algo", type=str, default="IP", help="the method for updating Q and AR filter")
    args = parser.parse_args()

    if args.gpu < 0:
        import numpy as xp
        print("Use CPU")
    else:
        try:
            import cupy as xp

            print("Use GPU " + str(args.gpu))
            xp.cuda.Device(args.gpu).use()
        except ImportError:
            print("Warning: cupy is not installed. 'gpu' argument should be set to -1. Switched to CPU.\n")
            import numpy as xp

    separater = AR_FastMNMF2(
        n_source=args.n_source,
        n_basis=args.n_basis,
        xp=xp,
        SCM=args.SCM,
        n_tap_AR=args.n_tap_AR,
        n_delay_AR=args.n_delay_AR,
        n_bit=args.n_bit,
        algo=args.algo,
        n_iter_init=args.n_iter_init
    )

    wav, sample_rate = sf.read(args.input_fname)
    wav /= np.abs(wav).max() * 1.2
    M = min(len(wav), args.n_mic)
    spec_FTM = MultiSTFT(wav[:, :M], n_fft=args.n_fft)

    separater.file_id = args.file_id
    separater.load_spectrogram(spec_FTM, sample_rate)
    separater.solve(
        n_iter=args.n_iter,
        save_dir="./",
        save_likelihood=False,
        save_param=False,
        save_wav=True,
        interval_save=5,
    )
