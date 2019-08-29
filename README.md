# SpeechEnhancement
Tools for multi-channel speech enhancement (and source separation)
* FullRank_Model includes 3 types of method called FCA, MNMF, MNMF-DP (MNMF with a deep prior)
* FastFullRank_Model includes 3 types of methods called FastFCA, FastMNMF, and FastMNMF-DP (FastMNMF with a deep prior) with iterative-projection (IP) method. These methods are extension of FCA, MNMF, MNMF-DP, respectively.
  - FastFCA is a method for general source separation. In fact, it can be available only for speech enhancement because of the strong initial value dependency.
  - FastMNMF is a general source separation method which integrate NMF-based source model into FastFCA.
  - FastMNMF-DP is a method which integrates deep speech prior into FastMNMF, and is for speech enhancement.
* We plan to add rank-1 model (ILRMA, and ILRMA-DP) in the future.

## Requirement
* Tested on Python3.6
* numpy
* pickle
* librosa
* soundfile
* progressbar2
* chainer (6.1.0 was tested) (for MNMF-DP, FastMNMF-DP, ILRMA-DP)
* cupy (6.1.0 was tested) (for GPU accelaration)

## Usage
```
python3 FastMNMF.py [input_filename] --gpu [gpu_id]
```
Input is the multichannel observed signals.  
If gpu_id < 0, CPU is used, and cupy is not necessary.


## Citation
If you use my code in a research project, please cite the following paper:

Kouhei Sekiguchi, Aditya Arie Nugraha, Yoshiaki Bando, Kazuyoshi Yoshii:  
[Fast Multichannel Source Separation Based on Jointly Diagonalizable Spatial Covariance Matrices](https://arxiv.org/abs/1903.03237),  
arXiv preprint arXiv:1903.03237, 2019
