# Jointly Diagonalizable Full-rank Model
## Introduction
- FastFCA.py
- FastMNMF.py
- FastMNMF_DP.py

FastMNMF is an extension of FastFCA.  
Since FastFCA has inter-frequency permutation problem, 
please use FastMNMF for source separation (and speech enhancement).  
In FastMNMF_DP, a deep generative model trained by using only clean speech
is used instead of NMF-based speech model in FastMNMF.  
Although FastMNMF_DP.py is only for speech enhancement, 
it is possible to extend this implementation for a multi speakers mixture.

## Usage
```
python3 FastMNMF.py <input filename> <options (e.g. --n_source 3)>
```
### Options
+ n_source : the number of sound sources.
+ n_iteration : the number of iterations to update parameters.
+ n_basis : the number of bases of NMF-based source model (n_basis_noise in FastMNMF_DP.py)
+ gpu : the index of GPU that you use. '-1' indicates CPU.
+ init_SCM : how to initialize the spatial model.
  - circular
  - gradual (recommendation for source separation)
  - obs (recommendation for speech enhancement)
  - ILRMA
+ n_bit : the number of bits of floating point number (32 or 64).  
By using 32, you can reduce computational cost and memory usage in exchange for the performance. 
Especially when the number of microphones is large, the performance is likely to degrade. 
Moreover, when you are using simulated signals without reverberation, 
since the mixture SCM is likely to be rank-deficient, 
please add some noise to the simulated signals (as in FastMNMF.py).
