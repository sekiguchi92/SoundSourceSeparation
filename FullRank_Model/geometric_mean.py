#! /usr/bin/env python3
# coding: utf-8

import sys, os
import numpy as np

sys.path.append("../CupyLibrary")
try:
    from chainer import cuda
    import cupy as cp
    FLAG_GPU_Available = True
except:
    print("---Warning--- You cannot use GPU acceleration because chainer or cupy is not installed")
    FLAG_GPU_Available = False

try:
    from cupy_matrix_inverse import inv_gpu_batch
    FLAG_CupyInverse_Enabled = True
except:
    print("---Warning--- You cannot use cupy inverse calculation")
    FLAG_CupyInverse_Enabled = False

try:
    import cupy_eig
    FLAG_CupyDeterminant_Enabled = True
except:
    print("---Warning--- You cannot use cupy complex determinant")
    FLAG_CupyDeterminant_Enabled = False


def matrix_sqrt(A):
    eig_val, eig_vec = np.linalg.eigh(A)
    if eig_val.ndim == 1:
        A_sqrt = eig_vec @ np.diag(np.sqrt(eig_val)) @ eig_vec.T.conj()
    else:
        tmp = np.zeros_like(A)
        M = A.shape[-1]
        eig_val = np.sqrt(eig_val)
        if eig_val.ndim == 2:
            for m in range(M):
                tmp[:, m, m] = eig_val[:, m]
            transpose_index = [0, 2, 1]
        elif eig_val.ndim == 3:
            for m in range(M):
                tmp[:, :, m, m] = eig_val[:, :, m]
            transpose_index = [0, 1, 3, 2]
        else:
            print("Error: Matrix size is too big. Please rewrite geometric_mean.py.")
            raise NotImplementedError
        A_sqrt = eig_vec @ tmp @ eig_vec.transpose(transpose_index).conj()
    return A_sqrt


def matrix_sqrt_for_cupy_HermitianMatrix(A):
    eig_val, eig_vec = cupy_eig.eigh(A, with_eigen_vector=True, upper_or_lower=False)
    if eig_val.ndim == 1:
        A_sqrt = eig_vec @ cp.diag(cp.sqrt(eig_val)) @ eig_vec.T.conj()
    else:
        tmp = cp.zeros_like(A)
        M = A.shape[-1]
        eig_val = cp.sqrt(eig_val)
        if eig_val.ndim == 2:
            for m in range(M):
                tmp[:, m, m] = eig_val[:, m]
            transpose_index = [0, 2, 1]
        if eig_val.ndim == 3:
            for m in range(M):
                tmp[:, :, m, m] = eig_val[:, :, m]
            transpose_index = [0, 1, 3, 2]
        A_sqrt = eig_vec @ tmp @ eig_vec.transpose(transpose_index).conj()
    return A_sqrt


def geometric_mean(A, B, xp=np):
    if xp == np:
        A_half = matrix_sqrt(A)
        A_half_inv = np.linalg.inv(A_half)
        ans = A_half @ matrix_sqrt(A_half_inv @ B @ A_half_inv) @ A_half
    else:
        A_half = matrix_sqrt_for_cupy_HermitianMatrix(A)
        A_half_inv = inv_gpu_batch(A_half)
        ans = A_half @ matrix_sqrt_for_cupy_HermitianMatrix(A_half_inv @ B @ A_half_inv) @ A_half
    return ans


def geometric_mean_invA(A_inv, B, xp=np):
    if xp == np:
        A_half_inv = matrix_sqrt(A_inv)
        A_half = np.linalg.inv(A_half_inv)
        ans = A_half @ matrix_sqrt(A_half_inv @ B @ A_half_inv) @ A_half
    elif FLAG_CupyDeterminant_Enabled:
        A_half_inv = matrix_sqrt_for_cupy_HermitianMatrix(A_inv)
        A_half = inv_gpu_batch(A_half_inv)
        ans = A_half @ matrix_sqrt_for_cupy_HermitianMatrix(A_half_inv @ B @ A_half_inv) @ A_half
    else:
        A_half_inv = matrix_sqrt(cuda.to_cpu(A_inv))
        A_half = np.linalg.inv(A_half_inv)
        ans = cuda.to_gpu(A_half @ matrix_sqrt(A_half_inv @ cuda.to_cpu(B) @ A_half_inv) @ A_half)
    return ans
