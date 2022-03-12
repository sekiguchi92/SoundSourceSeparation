#! /usr/bin/env python3
# coding: utf-8

import sys, os
import torch
import torchaudio
from pathlib import Path

sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))
from Base import EPS, MIC_INDEX, Base, MultiSTFT


class MNMF(Base):
    """The blind source separation using MNMF

    X_FTM: the observed complex spectrogram
    SCM_NFMM: spatial covariance matrices (SCMs) for each source
    W_NFK: basis vectors
    H_NKT: activations
    PSD_NFT: power spectral densities (PSDs) of each source (W_NFK @ H_NKT)
    """

    def __init__(self, n_source=2, n_basis=2, init_SCM="unit", n_iter_init=15, interval_norm=10, device="cpu", seed=0):
        """initialize MNMF

        Parameters:
        -----------
            n_source: int
                The number of sources.
            n_basis: int
                The number of bases of each source.
            init_SCM: str
                How to initialize SCM {unit, obs, ILRMA}.
                'obs' is for the case that one speech is dominant in the mixture.
            n_iter_init: int
                The number of iteration for ILRMA in 'ILRMA' initialization.
        """
        super(MNMF, self).__init__(device=device, seed=seed, n_bit=64)
        self.save_param_list += ["W_NFK", "H_NKT", "SCM_NFMM"]
        self.n_source = n_source
        self.n_basis = n_basis
        self.init_SCM = init_SCM
        self.n_iter_init = n_iter_init
        self.interval_norm = interval_norm
        self.method_name = "MNMF"

    def __str__(self):
        filename_suffix = (
            f"M={self.n_mic}-S={self.n_source}-F={self.n_freq}-K={self.n_basis}"
            f"-init={self.init_SCM}-bit={self.n_bit}-intv_norm={self.interval_norm}"
        )
        if hasattr(self, "file_id"):
            filename_suffix += f"-ID={self.file_id}"
        return filename_suffix

    def init_source_model(self):
        self.W_NFK = torch.rand(self.n_source, self.n_freq, self.n_basis, dtype=self.TYPE_FLOAT, device=self.device)
        self.H_NKT = torch.rand(self.n_source, self.n_basis, self.n_time, dtype=self.TYPE_FLOAT, device=self.device)

    def init_spatial_model(self):
        self.SCM_NFMM = torch.tile(
            torch.eye(self.n_mic, dtype=self.TYPE_COMPLEX, device=self.device), [self.n_source, self.n_freq, 1, 1]
        )
        if "unit" in self.init_SCM:
            pass
        elif "obs" in self.init_SCM:  # Mainly For speech enhancement
            self.XX_FTMM = torch.einsum("fti, ftj -> ftij", self.X_FTM, self.X_FTM.conj())
            ave_power_FT = (torch.abs(self.X_FTM) ** 2).mean(axis=2)
            self.SCM_NFMM[0] = torch.einsum("fij, f -> fij", self.XX_FTMM.sum(axis=1), ave_power_FT.sum(axis=1))
        elif "ilrma" in self.init_SCM.lower():
            from ILRMA import ILRMA

            ilrma = ILRMA(n_basis=2, init_SCM="unit", device=self.device)
            ilrma.load_spectrogram(self.X_FTM)
            ilrma.solve(n_iter=self.n_iter_init, save_wav=False)
            self.start_idx = self.n_iter_init
            MixingMatrix_FMM = torch.linalg.inv(ilrma.Q_FMM)
            separated_spec_power = torch.abs(ilrma.separated_spec).mean(axis=(1, 2))
            for n in range(self.n_source):
                self.SCM_NFMM[n] = torch.einsum(
                    "fi, fj -> fij",
                    MixingMatrix_FMM[..., separated_spec_power.argmax()],
                    MixingMatrix_FMM[..., separated_spec_power.argmax()].conj(),
                )
                separated_spec_power[separated_spec_power.argmax()] = -torch.inf
            self.SCM_NFMM /= torch_trace(self.SCM_NFMM).real[:, :, None, None]
            self.SCM_NFMM += 1e-2 * torch.eye(self.n_mic, device=self.device)[None, None]
        elif "fastmnmf" in self.init_SCM.lower():
            from FastMNMF2 import FastMNMF2

            fastmnmf2 = FastMNMF2(n_source=self.n_source, n_basis=2, init_SCM="circular", device=self.device)
            fastmnmf2.load_spectrogram(self.X_FTM)
            fastmnmf2.solve(n_iter=self.n_iter_init, save_wav=False)
            self.start_idx = self.n_iter_init
            Qinv_FMM = torch.linalg.inv(fastmnmf2.Q_FMM)
            self.SCM_NFMM = torch.einsum("fij, nj, fkj -> nfik", Qinv_FMM, fastmnmf2.G_NM, Qinv_FMM.conj())

        self.normalize()

    def calculate_PSD(self):
        self.PSD_NFT = self.W_NFK @ self.H_NKT + EPS

    def update(self):
        self.update_axiliary_variable()
        self.update_WH()
        self.update_SCM()
        self.normalize()

    def update_axiliary_variable(self):
        self.Yinv_FTMM = torch.linalg.solve(
            torch.einsum("nft, nfij -> ftij", self.PSD_NFT.to(self.TYPE_COMPLEX), self.SCM_NFMM),
            torch.eye(self.n_mic, dtype=self.TYPE_COMPLEX, device=self.device)[None, None],
        )

        Yinv_x_FTM = torch.einsum("ftij, ftj -> fti", self.Yinv_FTMM, self.X_FTM)
        self.Yinv_X_Yinv_FTMM = torch.einsum("fti, ftj -> ftij", Yinv_x_FTM, Yinv_x_FTM.conj())
        self.tr_SCM_Yinv_X_Yinv_NFT = torch_trace(
            torch.einsum("nfij, ftjl -> nftil", self.SCM_NFMM, self.Yinv_X_Yinv_FTMM)
        ).real
        self.tr_SCM_Yinv_NFT = torch_trace(torch.einsum("nfij, ftjl -> nftil", self.SCM_NFMM, self.Yinv_FTMM)).real

    def update_WH(self):
        W_numerator_NFK = torch.einsum("nkt, nft -> nfk", self.H_NKT, self.tr_SCM_Yinv_X_Yinv_NFT)
        W_denominator_NFK = torch.einsum("nkt, nft -> nfk", self.H_NKT, self.tr_SCM_Yinv_NFT)

        H_numerator_NKT = torch.einsum("nfk, nft -> nkt", self.W_NFK, self.tr_SCM_Yinv_X_Yinv_NFT)
        H_denominator_NKT = torch.einsum("nfk, nft -> nkt", self.W_NFK, self.tr_SCM_Yinv_NFT)

        self.W_NFK *= torch.sqrt(W_numerator_NFK / W_denominator_NFK)
        self.H_NKT *= torch.sqrt(H_numerator_NKT / H_denominator_NKT)

    def update_SCM(self):
        left_NFMM = torch.einsum("nft, ftij -> nfij", self.PSD_NFT.to(self.TYPE_COMPLEX), self.Yinv_FTMM)
        b = torch.einsum("nft, ftij -> nfij", self.PSD_NFT.to(self.TYPE_COMPLEX), self.Yinv_X_Yinv_FTMM)
        right_NFMM = (
            torch.einsum("nfij, nfjk, nfkl -> nfil", self.SCM_NFMM, b, self.SCM_NFMM)
            + (torch.eye(self.n_mic, dtype=self.TYPE_COMPLEX, device=self.device) * EPS)[None, None]
        )
        self.SCM_NFMM = geometric_mean_Ainv(left_NFMM, right_NFMM)
        self.SCM_NFMM = (self.SCM_NFMM + self.SCM_NFMM.permute(0, 1, 3, 2).conj()) / 2

    def normalize(self):
        mu_NF = torch_trace(self.SCM_NFMM).real
        self.SCM_NFMM /= mu_NF[:, :, None, None]
        self.W_NFK *= mu_NF[:, :, None]

        nu_NK = self.W_NFK.sum(axis=1)
        self.W_NFK /= nu_NK[:, None]
        self.H_NKT *= nu_NK[:, :, None]

        self.calculate_PSD()

    def separate(self, mic_index=MIC_INDEX):
        Omega_NFTMM = self.PSD_NFT[:, :, :, None, None] * self.SCM_NFMM[:, :, None]
        Omega_sum_inv_FTMM = torch.linalg.inv(Omega_NFTMM.sum(axis=0))
        self.separated_spec = torch.einsum("nftij, ftjk, ftk -> nfti", Omega_NFTMM, Omega_sum_inv_FTMM, self.X_FTM)[
            ..., mic_index
        ]
        return self.separated_spec

    def calculate_log_likelihood(self):
        if not hasattr(self, "XX_FTMM"):
            self.XX_FTMM = torch.einsum("fti, ftj -> ftij", self.X_FTM, self.X_FTM.conj())
        Y_FTMM = torch.einsum("nft, nfij -> ftij", self.PSD_NFT.to(self.TYPE_COMPLEX), self.SCM_NFMM)
        return (-torch_trace(torch.linalg.inv(Y_FTMM) @ self.XX_FTMM).real).sum() - torch.log(
            torch.linalg.det(Y_FTMM)
        ).sum().real

    def load_param(self, filename):
        super().load_param(filename)

        self.n_source, self.n_freq, self.n_basis = self.W_NFK.shape
        _, _, self.n_time = self.H_NKT


def torch_trace(mat):
    return torch.diagonal(mat, dim1=-2, dim2=-1).sum(axis=-1)


def matrix_sqrth(A_NFMM):
    eig_val_NFM, eig_vec_NFMM = torch.linalg.eigh(A_NFMM)
    eig_val_NFM[eig_val_NFM < EPS] = EPS
    return torch.einsum("nfij, nfj, nflj -> nfil", eig_vec_NFMM, torch.sqrt(eig_val_NFM), eig_vec_NFMM.conj())


def geometric_mean_Ainv(Ainv_NFMM, B_NFMM):
    Asqrt_inv = matrix_sqrth(Ainv_NFMM)
    Asqrt = torch.linalg.inv(Asqrt_inv)
    return Asqrt @ matrix_sqrth(Asqrt_inv @ B_NFMM @ Asqrt_inv) @ Asqrt


if __name__ == "__main__":
    import soundfile as sf
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_fname", type=str, help="filename of the multichannel observed signals")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--n_fft", type=int, default=1024, help="number of frequencies")
    parser.add_argument("--n_source", type=int, default=3, help="number of noise")
    parser.add_argument("--n_basis", type=int, default=16, help="number of basis")
    parser.add_argument(
        "--init_SCM", type=str, default="FastMNMF2", help="unit, obs (onny for enhancement), ILRMA, or FastMNMF2"
    )
    parser.add_argument("--n_iter", type=int, default=100, help="number of iteration")
    parser.add_argument("--n_mic", type=int, default=8, help="number of microphone")
    args = parser.parse_args()

    if args.gpu < 0:
        device = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = "cuda"

    separater = MNMF(
        n_source=args.n_source,
        n_basis=args.n_basis,
        device=device,
        init_SCM=args.init_SCM,
    )

    wav, sample_rate = torchaudio.load(args.input_fname, channels_first=False)
    wav /= torch.abs(wav).max() * 1.2
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
