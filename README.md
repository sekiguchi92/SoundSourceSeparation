# Sound Source Separation
Tools for multi-channel sound source separation and dereverberation.

## News
* I published new paper entitled ["Autoregressive Moving Average Jointly-Diagonalizable Spatial Covariance Analysis for Joint Source Separation and Dereverberation, IEEE/ACM TASLP2022"](https://ieeexplore.ieee.org/abstract/document/9829286).
* FastBSSD.py performs joint source separation and dereverberation. You can select speech model from NMF, DNN, and FreqInv models, noise model from NMF and TimeInv models, and reverberation model from AR, MA, ARMA, and None. Please check my paper for details.
* I also published another paper entitled ["Direction-aware adaptive online neural speech enhancement with an augmented reality headset in real noisy conversational environments, IROS2022](https://arxiv.org/abs/2207.07296).

## Method list
### Source separation
* FastMNMF1
* FastMNMF2
* FastMNMF2_DP (DNN speech model + NMF noise model)
* FastBSS2 (Frequency invariant / NMF / DNN speech model + NMF / Time invariant noise model)
  - This method includes FastMNMF2, FastMNMF2_DP, and so on
* ILRMA
* MNMF (Pytorch version is much slower than cupy version on GPU)

### Joint source separation and dereverberation
* AR-FastMNMF2 (Pytorch version is not ready)

## Requirements
* Tested on Python3.8  
* Requirements for numpy and cupy version in `src` are listed below
```
numpy (1.19.2 was tested)
librosa
pysoundfile
tqdm

# optional packages
cupy # for GPU accelaration (9.4.0 was tested)
h5py # for saving the estimated parameters
torch # for using DNN source model in FastBSS2.py or FastMNMF2_DP.py
```
You can install all the packages above with `pip install -r src/requirements.txt`  

* Requirements for pytorch version in `src_torch` are listed below
```
torch
torchaudio
tqdm

# optional packages
h5py # for saving the estimated parameters
```
You can install all the packages above with `pip install -r src_torch/requirements.txt`  

## Usage
```
python3 FastMNMF2.py [input_filename] --gpu [gpu_id]
```
* Input is the multichannel observed signals.  
* If gpu_id < 0, CPU is used, and cupy is not required.


## Citation
If you use the code of FastMNMF1 or FastMNMF2 in your research project, please cite the following paper:

* Kouhei Sekiguchi, Aditya Arie Nugraha, Yoshiaki Bando, Kazuyoshi Yoshii:  
 [Fast Multichannel Source Separation Based on Jointly Diagonalizable Spatial Covariance Matrices](https://ieeexplore.ieee.org/abstract/document/8902557),  
 European Signal Processing Conference (EUSIPCO), 2019
* Kouhei Sekiguchi, Yoshiaki Bando, Aditya Arie Nugraha, Kazuyoshi Yoshii, Tatsuya Kawahara:  
[Fast Multichannel Nonnegative Matrix Factorization with Directivity-Aware Jointly-Diagonalizable Spatial Covariance Matrices for Blind Source Separation](https://ieeexplore.ieee.org/document/9177266),  
IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2020.  

If you use the code of AR-FastMNMF2 in your research project, please cite the following paper:

* Kouhei Sekiguchi, Yoshiaki Bando, Aditya Arie Nugraha, Mathieu Fontaine, Kazuyoshi Yoshii:
[Autoregressive Fast Multichannel Nonnegative Matrix Factorization for Joint Blind Source Separation and Dereverberation](https://ieeexplore.ieee.org/document/9414857),
IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021.

<!-- If you use the code of "FullRank Model" in a research project, please cite the following paper:  
* Kouhei Sekiguchi, Yoshiaki Bando, Aditya Arie Nugraha, Kazuyoshi Yoshii, Tatsuya Kawahara:  
  [Semi-supervised Multichannel Speech Enhancement with a Deep Speech Prior](https://ieeexplore.ieee.org/document/8861142),  
  IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol 27, no 12, pp. 2197-2212, 2019
* Kouhei Sekiguchi, Yoshiaki Bando, Kazuyoshi Yoshii, Tatsuya Kawahara:  
  [Bayesian Multichannel Speech Enhancement with a Deep Speech Prior](http://www.apsipa.org/proceedings/2018/pdfs/0001233.pdf),  
  Asia-Pacific Signal and Information Processing Association (APSIPA), 2018
 
 If you use the code of "Rank1 Model" in a research project, please cite the following paper:  
* Kouhei Sekiguchi, Yoshiaki Bando, Aditya Arie Nugraha, Kazuyoshi Yoshii, Tatsuya Kawahara:  
  [Semi-supervised Multichannel Speech Enhancement with a Deep Speech Prior](https://ieeexplore.ieee.org/document/8861142),  
  IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol 27, no 12, pp. 2197-2212, 2019
 -->

## Detail
- "n_bit" argument means the number of bits, and is set to 32 or 64 (32-> float32 and complex64, 64->float64 and complex128, default is 64).
n_bit=32 reduces computational cost and memory usage in exchange for the separation performance.
Especially when the number of microphones (or tap length in AR-based methods like AR-FastMNMF2) is large, the performance is likely to degrade.
Moreover, when you are using simulated signals without reverberation, since the mixture SCM is likely to be rank-deficient,
please add small noise to the simulated signals. In MNMF.py, only n_bit=64 is available.

