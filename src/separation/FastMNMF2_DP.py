#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import sys, os
import soundfile as sf

import torch

from pathlib import Path

sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))
from FastBSS2 import MultiSTFT, FastBSS2

"""
Sound source separation method with DNN speech model, NMF noise model, and
weight-shared jointly-diagonalizable (WJD) spatial model, which is used in FastMNMF2.
This is a special case of FastBSS2.py
"""

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_fname", type=str, help="filename of the multichannel observed signals")

    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--n_speech", type=int, default=3, help="number of speech")
    parser.add_argument("--n_noise", type=int, default=0, help="number of noise")
    parser.add_argument("--n_iter", type=int, default=100, help="number of iteration")
    parser.add_argument("--algo", type=str, default="IP", help="IP or ISS")

    # Only for NMF source model
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

    # Start separation
    separater = FastBSS2(
        speech_model="DNN", noise_model="NMF", init_SCM="twostep", speech_VAE=speech_VAE, xp=xp, **vars(args)
    )
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
