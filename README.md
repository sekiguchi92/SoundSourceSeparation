# Sound Source Separation
Tools for multi-channel sound source separation and dereverberation.

## News
* Ver2.0 is released. The code of FastMNMF1 is refactored and FastMNMF2 and AR-FastMNMF2 are newly added.
* AR-FastMNMF2 is the extension of FastMNMF2 for joint blind source separation and dereverberation.
* Other methods implemented at ver1.0 such as ILRMA, MNMF, MNMF-DP, and FastMNMF-DP will be added in the future.

## Method list
### Source separation
* FastMNMF1
* FastMNMF2

### Joint source separation and dereverberation
* AR-FastMNMF2

## Requirements
* Tested on Python3.8  
* Minimal requirements are listed below
```
numpy (1.19.2 was tested)
librosa
pysoundfile
tqdm
```
You can install all the packages above with `pip install -r requirements.txt`  
  
* Optional packages are listed below
```
cupy # for GPU accelaration (9.4.0 was tested)
h5py # for saving the estimated parameters
```

## Usage
```
python3 FastMNMF.py [input_filename] --gpu [gpu_id]
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
please add small noise to the simulated signals.

