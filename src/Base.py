#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

EPS = 1e-10
MIC_INDEX = 0


def MultiSTFT(wav_TM: "np.ndarray", n_fft=1024, hop_length=None) -> np.ndarray:
    """
    Multichannel STFT

    Parameters
    ---------
    wav_TM: ndarray (T x M) or (T)
    n_fft: int
        The window size (default 1024)
    hop_length: int
        The shift length (default None)
        If None, n_fft // 4

    Returns
    -------
    spec_FTM: np.ndarray (F x T x M) or (F x T)
    """
    if hop_length is None:
        hop_length = n_fft // 4

    if wav_TM.ndim == 1:
        wav_TM = wav_TM[:, None]

    _, M = wav_TM.shape

    for m in range(M):
        spec = librosa.core.stft(wav_TM[:, m], n_fft=n_fft, hop_length=hop_length)
        if m == 0:
            spec_FTM = np.zeros([*spec.shape, M], dtype=spec.dtype)
        spec_FTM[..., m] = spec

    return spec_FTM.squeeze()


def MultiISTFT(spec, hop_length=None, shape="FTM"):
    """
    Multichannel inverse STFT

    Parameters
    ---------
    spec: np.ndarray (F x T x M) or (F x T)
    hop_length: int
        The shift length (default None)
        If None, (F-1) * 4
    shape: str
        Shape of the spec. FTM or MFT

    Returns
    -------
    wav_TM: np.ndarray ((T' x M) or T')
    """
    if spec.ndim == 2:
        spec = spec[..., None]
        shape = "FTM"

    if shape == "MFT":
        spec = spec.transpose(1, 2, 0)

    F, _, M = spec.shape

    if hop_length is None:
        n_fft = (F - 1) * 2
        hop_length = n_fft // 4

    for m in range(M):
        wav = librosa.core.istft(spec[..., m], hop_length=hop_length)
        if m == 0:
            wav_TM = np.zeros([len(wav), M], dtype=wav.dtype)
        wav_TM[:, m] = wav

    return wav_TM.squeeze()


class Base:
    " Base Class for Source Separation Methods"

    def __init__(self, xp=np, seed=0, n_bit=64):
        self.xp = xp
        np.random.seed(seed)
        self.xp.random.seed(seed)

        self.n_bit = n_bit
        if self.n_bit == 64:
            self.TYPE_FLOAT = self.xp.float64
            self.TYPE_COMPLEX = self.xp.complex128
        elif self.n_bit == 32:
            self.TYPE_FLOAT = self.xp.float32
            self.TYPE_COMPLEX = self.xp.complex64
        self.method_name = "Base"
        self.save_param_list = ["n_bit"]

    def convert_to_NumpyArray(self, data):
        if self.xp == np:
            return data
        else:
            return self.xp.asnumpy(data)

    def load_spectrogram(self, X_FTM, sample_rate=16000):
        """ load complex spectrogram

        Parameters:
        -----------
            X_FTM: np.ndarray F x T x M
                Spectrogram of observed signals
        """
        self.n_freq, self.n_time, self.n_mic = X_FTM.shape
        self.X_FTM = self.xp.asarray(X_FTM, dtype=self.TYPE_COMPLEX)
        self.sample_rate = sample_rate
        self.start_idx = 0

    def solve(
        self,
        n_iter=100,
        save_dir="./",
        save_wav=True,
        save_wav_all=False,
        save_param=False,
        save_param_all=False,
        save_likelihood=False,
        interval_save=30,
        mic_index=MIC_INDEX,
        init=True,
    ):
        """
        Parameters:
            n_iter: int
            save_dir: str
            save_wav: bool
                Save the separated signals only after the last iteration 
            save_wav_all: bool
                Save the separated signals at every 'interval_save' iterations
            save_param: bool
            save_param_all: bool
            save_likelihood: bool
            interval_save: int
                interval of saving wav, parameter, and log-likelihood
        """
        self.n_iter = n_iter
        if init:
            self.init_source_model()
            self.init_spatial_model()

        print(f"Update {self.method_name}-{self}  {self.n_iter-self.start_idx} times ...")

        self.log_likelihood_dict = {}
        for self.it in tqdm(range(self.start_idx, n_iter)):
            self.update()

            if save_param_all and ((self.it + 1) % interval_save == 0) and ((self.it + 1) != n_iter):
                save_fname = f"{save_dir}/{self.method_name}-param-{str(self)}-{self.it+1}.h5"
                self.save_param(save_fname)

            if save_wav_all and ((self.it + 1) % interval_save == 0) and ((self.it + 1) != n_iter):
                self.separate(mic_index=mic_index)
                save_fname = f"{save_dir}/{self.method_name}-sep-{str(self)}-{self.it+1}.wav"
                self.save_to_wav(self.separated_spec, save_fname=save_fname, shape="FTM")

            if save_likelihood and ((self.it + 1) % interval_save == 0) and ((self.it + 1) != n_iter):
                self.log_likelihood_dict[self.it + 1] = float(self.calculate_log_likelihood())

        self.separate(mic_index=mic_index)
        if save_wav or save_wav_all:
            save_fname = f"{save_dir}/{self.method_name}-sep-{str(self)}.wav"
            self.save_to_wav(self.separated_spec, save_fname=save_fname, shape="FTM")

        if save_param or save_param_all:
            save_fname = f"{save_dir}/{self.method_name}-param-{str(self)}.h5"
            self.save_param(save_fname)

        if save_likelihood:
            self.log_likelihood_dict[n_iter] = float(self.calculate_log_likelihood())
            save_fname = f"{save_dir}/{self.method_name}-ll-{str(self)}.txt"
            with open(save_fname, "w") as f:
                for key, val in self.log_likelihood_dict.items():
                    f.write(f"it = {key} : log_likelihood = {val}\n")

    def save_to_wav(self, spec, save_fname="./sample.wav", shape="FTM"):
        spec = self.convert_to_NumpyArray(spec)
        assert not np.isnan(spec).any(), "spec includes NaN"
        assert spec.ndim <= 3, f"shape of spec is wrong : {spec.shape}"

        separated_signal = MultiISTFT(spec.transpose(1, 2, 0), shape=shape)
        sf.write(save_fname, separated_signal, self.sample_rate)

    def save_param(self, fname):
        import h5py
        with h5py.File(fname, "w") as f:
            for param in self.save_param_list:
                data = getattr(self, param)
                if type(data) is self.xp.ndarray:
                    data = self.convert_to_NumpyArray(data)
                f.create_dataset(param, data=data)
            f.flush()

    def load_param(self, fname):
        import h5py
        with h5py.File(fname, "r") as f:
            for key in f.keys():
                data = f[key]
                if (type(data) is self.xp) and (self.xp is not np):
                    data = self.xp.asarray(data)
                setattr(self, key, data)

            if "n_bit" in f.keys():
                if self.n_bit == 64:
                    self.TYPE_COMPLEX = self.xp.complex128
                    self.TYPE_FLOAT = self.xp.float64
                else:
                    self.TYPE_COMPLEX = self.xp.complex64
                    self.TYPE_FLOAT = self.xp.float32
