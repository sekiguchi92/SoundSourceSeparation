# SpeechEnhancement
Tools for multi-channel speech enhancement (and source separation)

## FullRank Model
FullRank_Model includes 3 methods called Full-rank Spatial Covariance Analysis (FCA), Multichannel Nonnegative Matrix Factorization(MNMF), MNMF with a deep prior (MNMF-DP).  
These methods are based on full-rank spatial model.

 * FCA is a method for general source separation. In fact, it can be available only for speech enhancement because of the strong initial value dependency.
 * MNMF is a general source separation method which integrate NMF-based source model into FCA.
 * MNMF-DP is a method which integrates deep speech prior into MNMF, and is only for speech enhancement.
 
 
## Jointly Diagonalizable FullRank Model
Jointly_Diagonalizable_FullRank_Model includes 3 methods called FastFCA, FastMNMF, and FastMNMF-DP (FastMNMF with a deep prior) with iterative-projection (IP) method.   
These methods are based on the jointly diagonalizable full-rank spatial model, and they are extension of FCA, MNMF, MNMF-DP, respectively.

  - FastFCA is a method for general source separation. In fact, it can be available only for speech enhancement because of the strong initial value dependency.
  - FastMNMF is a general source separation method which integrate NMF-based source model into FastFCA.
  - FastMNMF-DP is a method which integrates deep speech prior into FastMNMF, and is only for speech enhancement.
  
  
## Rank-1 Model
Rank1_Model includes 2 methods called Independent Low-Rank Matrix Analysis (ILRMA) and ILRMA with a deep prior (ILRMA-DP).  
These methods are based on rank-1 spatial model.
 * ILRMA is a general source separation method which integrate NMF-based source model into rank-1 spatial model.
 * ILRMA-DP is a method which integrates deep speech prior into ILRMA, and is only for speech enhancement.


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
If you use the code of "Jointly Diagonalizable FullRank Model" in a research project, please cite the following paper:

Kouhei Sekiguchi, Aditya Arie Nugraha, Yoshiaki Bando, Kazuyoshi Yoshii:  
[Fast Multichannel Source Separation Based on Jointly Diagonalizable Spatial Covariance Matrices](https://www.semanticscholar.org/paper/Fast-Multichannel-Source-Separation-Based-on-Sekiguchi-Nugraha/d754917d0cdd127a681713d82a378861c5507faf),  
European Signal Processing Conference (EUSIPCO), 2019

If you use the code of "FullRank Model" in a research project, please cite the following paper:
* Kouhei Sekiguchi, Yoshiaki Bando, Aditya Arie Nugraha, Kazuyoshi Yoshii, Tatsuya Kawahara:
  [Semi-supervised Multichannel Speech Enhancement with a Deep Speech Prior](https://ieeexplore.ieee.org/document/8861142),
  IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol 27, no 12, pp. 2197-2212, 2019
* Kouhei Sekiguchi, Yoshiaki Bando, Kazuyoshi Yoshii, Tatsuya Kawahara:
  [Bayesian Multichannel Speech Enhancement with a Deep Speech Prior](http://www.apsipa.org/proceedings/2018/pdfs/0001233.pdf),
  Asia-Pacific Signal and Information Processing Association (APSIPA), 2018