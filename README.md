# SpeechEnhancement
Tools for multi-channel speech enhancement (and source separation)
## Requirement
* Tested on Python3.6
* numpy
* pickle
* librosa
* soundfile
* progressbar2
* chainer (5.3.0 was tested) (for MNMF-DP, FastMNMF-DP, ILRMA-DP)
* cupy (5.3.0 was tested) (for GPU accelaration)

## Citation
If you use my code in a research project, please cite the following paper:

Kouhei Sekiguchi, Aditya Arie Nugraha, Yoshiaki Bando, Kazuyoshi Yoshii:
[Fast Multichannel Source Separation Based on Jointly Diagonalizable Spatial Covariance Matrices](https://arxiv.org/abs/1903.03237),
arXiv preprint arXiv:1903.03237, 2019
