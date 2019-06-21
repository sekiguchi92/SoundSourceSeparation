#! /usr/bin/env python3

import glob, os
import pickle as pic
import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf
import librosa
from progressbar import progressbar

from configure_VAE import *

def vad(amp_spec, W=5):
    sig = fftconvolve(amp_spec.mean(axis=0), np.ones(W) / W, 'same')
    th = max(abs(sig)) * 0.01
    return sig > th

def make_dataset(wsj0_path=WSJ0_PATH, dataset_save_path=DATASET_SAVE_PATH):
    dataset_fileName = dataset_save_path + '/wsj0_normalize_{}_{}.pic'.format(N_FFT, HOP_LENGTH)
    if os.path.isfile(dataset_fileName):
        print(dataset_fileName, " already exist. Skip this phase.")
        return 0

    print("Start making dataset ...")
    dataset = []
    for fileName in progressbar(glob.glob(wsj0_path + "/*.wav")):
        wav, _ = snd.read(fname)
        pwr_spec_FT  = np.abs(librosa.core.stft(wav, n_fft=args.n_fft, hop_length=args.hop_length)) ** 2
        vad_result = vad(pwr_spec_FT)
        pwr_spec_FT /= (pwr_spec_FT.sum(axis=0)[vad(pwr_spec_FT)]).mean()
        dataset.append(np.array(pwr_spec_FT, dtype=np.float32))

    dataset = np.hstack(dataset)

    print("Writing to pickle file ...")
    pic.dump(dataset, open(dataset_fileName, 'wb'), protocol=4)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(     '--n_fft', type=int, default=1024)
    parser.add_argument('--hop_length', type=int, default= 256)
    args = parser.parse_args()
    wsj0_path = "/n/rd25/mimura/corpus/CHiME3/data/audio/16kHz/isolated/tr05_org"
    dataset_save_path = '/n/sd2/sekiguchi/dataset/'

    make_dataset(wsj0_path, dataset_save_path)

