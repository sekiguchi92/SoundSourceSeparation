import numpy
from numpy import linalg
import sys

import cupy
from cupy.core import core
from cupy import cuda
from cupy.cuda import cublas
from cupy.cuda import device
from cupy.linalg import decomposition
from cupy.linalg import util

if cuda.cusolver_enabled:
    from cupy.cuda import cusolver
    # sys.path.append("/home/sekiguch/cupy-master/cupy/cuda/")
    from cupy.cuda import my_cusolver


def eigh(x, with_eigen_vector=True, upper_or_lower=True):
    """eigen value decomposition for Hermitian matrix

    Args:
        x (cupy.ndarray): The regular matrix
        with_eigen_vector: boolean
            whether the eigen vector is returned or not
        upper_or_lower: boolean
            "eigh" function is only for Hermitian matrix.
            So, to caculate eigen value, only upper or lower part of the matrix is necessary.
            When the input is the upper part of the matrix, this value is True
    Returns:
        cupy.ndarray: eigen values
        cupy.ndarray: eigen vectors (only when with_eigen_vector=True)
    """
    if not cuda.cusolver_enabled:
        raise RuntimeError('Current cupy only supports cusolver in CUDA 8.0')
    if x.shape[-2] != x.shape[-1]:
        raise ValueError

    # to prevent `a` to be overwritten
    shape_array = x.shape
    a = x.reshape(-1, shape_array[-2], shape_array[-1]).copy()

    n = a.shape[1]
    batchSize = len(a)
    info = cupy.empty(batchSize, dtype=numpy.int32)
    cusolver_handle = device.Device().cusolver_handle

    params = my_cusolver.DnCreateSyevjInfo(cusolver_handle)

    if a.dtype.char == 'f' or a.dtype.char == 'd' or a.dtype.char == 'F' or a.dtype.char == 'D':
        dtype = a.dtype.char
    else:
        # dtype = numpy.find_common_type((a.dtype.char, 'f'), ()).char
        print("Error: input dtype is not appropriate")
        raise ValueError

    if dtype == 'f':
        eigh = my_cusolver.DnSsyevjBatched
        eigh_bufferSize = my_cusolver.DnSsyevjBatched_bufferSize
        dtype_eig_val = 'f'
    elif dtype == 'd':
        eigh = my_cusolver.DnDsyevjBatched
        eigh_bufferSize = my_cusolver.DnDsyevjBatched_bufferSize
        dtype_eig_val = 'd'
    elif dtype == 'F':
        eigh = my_cusolver.DnCheevjBatched
        eigh_bufferSize = my_cusolver.DnCheevjBatched_bufferSize
        dtype_eig_val = 'f'
    elif dtype == 'D':
        eigh = my_cusolver.DnZheevjBatched
        eigh_bufferSize = my_cusolver.DnZheevjBatched_bufferSize
        dtype_eig_val = 'd'

    if with_eigen_vector: # 固有ベクトルも出すかどうか
        jobz = cusolver.CUSOLVER_EIG_MODE_VECTOR
    else:
        jobz = cusolver.CUSOLVER_EIG_MODE_NOVECTOR

    if upper_or_lower: # Hermitian行列だから，右上(upper)か左下(lower)のみ見れば良い．どちらを見るか
        uplo = cublas.CUBLAS_FILL_MODE_LOWER
    else:
        uplo = cublas.CUBLAS_FILL_MODE_UPPER

    eig_val = cupy.empty((batchSize, n), dtype=dtype_eig_val) # for eigen vector

    buffersize = eigh_bufferSize(cusolver_handle, jobz, uplo, n, a.data.ptr, n, eig_val.data.ptr, params, batchSize)
    workspace = cupy.empty(buffersize, dtype=dtype)

    # LU factorization
    eigh(cusolver_handle, jobz, uplo, n, a.data.ptr, n, eig_val.data.ptr, workspace.data.ptr, buffersize, info.data.ptr, params, batchSize)

    if batchSize == 1:
        return eig_val[0], a[0].T.conj()
    else:
        return eig_val.reshape(shape_array[:-1]), a.transpose(0, 2, 1).reshape(shape_array).conj()


def det_Hermitian(x, upper_or_lower=True):
    """eigen value decomposition for Hermitian matrix

    Args:
        x (cupy.ndarray): The regular matrix
        upper_or_lower: boolean
            "eigh" function is only for Hermitian matrix.
            So, to caculate eigen value, only upper or lower part of the matrix is necessary.
            When the input is the upper part of the matrix, this value is True
    Returns:
        cupy.ndarray: eigen values
    """
    if not cuda.cusolver_enabled:
        raise RuntimeError('Error : cusolver_enabled == False')
    if x.shape[-2] != x.shape[-1]:
        raise ValueError
    # to prevent `a` to be overwritten
    shape_array = x.shape
    a = x.reshape(-1, shape_array[-2], shape_array[-1]).copy()

    with_eigen_vector = False

    n = a.shape[1]
    batchSize = len(a)
    info = cupy.empty(batchSize, dtype=numpy.int32)
    cusolver_handle = device.Device().cusolver_handle

    params = my_cusolver.DnCreateSyevjInfo(cusolver_handle)

    if a.dtype.char == 'f' or a.dtype.char == 'd' or a.dtype.char == 'F' or a.dtype.char == 'D':
        dtype = a.dtype.char
    else:
        # dtype = numpy.find_common_type((a.dtype.char, 'f'), ()).char
        print("Error: input dtype is not appropriate")
        raise ValueError

    if dtype == 'f':
        eigh = my_cusolver.DnSsyevjBatched
        eigh_bufferSize = my_cusolver.DnSsyevjBatched_bufferSize
        dtype_eig_val = 'f'
    elif dtype == 'd':
        eigh = my_cusolver.DnDsyevjBatched
        eigh_bufferSize = my_cusolver.DnDsyevjBatched_bufferSize
        dtype_eig_val = 'd'
    elif dtype == 'F':
        eigh = my_cusolver.DnCheevjBatched
        eigh_bufferSize = my_cusolver.DnCheevjBatched_bufferSize
        dtype_eig_val = 'f'
    elif dtype == 'D':
        eigh = my_cusolver.DnZheevjBatched
        eigh_bufferSize = my_cusolver.DnZheevjBatched_bufferSize
        dtype_eig_val = 'd'

    jobz = cusolver.CUSOLVER_EIG_MODE_VECTOR

    if upper_or_lower: # Hermitian行列だから，右上(upper)か左下(lower)のみ見れば良い．どちらを見るか
        uplo = cublas.CUBLAS_FILL_MODE_LOWER
    else:
        uplo = cublas.CUBLAS_FILL_MODE_UPPER

    eig_val = cupy.empty((batchSize, n), dtype=dtype_eig_val) # for eigen vector

    buffersize = eigh_bufferSize(cusolver_handle, jobz, uplo, n, a.data.ptr, n, eig_val.data.ptr, params, batchSize)
    workspace = cupy.empty(buffersize, dtype=dtype)

    # LU factorization
    eigh(cusolver_handle, jobz, uplo, n, a.data.ptr, n, eig_val.data.ptr, workspace.data.ptr, buffersize, info.data.ptr, params, batchSize)

    if batchSize == 1:
        return eig_val[0].prod()
    else:
        eig_val = eig_val.prod(axis=1)
        return eig_val.reshape(shape_array[:-2])

if __name__ == "__main__":
    print("start")
    # a = cupy.random.rand(2, 2, 3, 3, dtype=cupy.float) + cupy.random.rand(2, 2, 3, 3, dtype=cupy.float) * 1j
    # a = a + a.transpose(0, 1, 3, 2).conj()
    # a = cupy.random.rand(2, 3, 3, dtype=cupy.float64) + cupy.random.rand(2, 3, 3, dtype=cupy.float64) * 1j
    # a = a + a.transpose(0, 2, 1).conj()
    a = cupy.random.rand(3, 3, dtype=cupy.float32)
    a = a + a.T.conj()
    a = a @ a
    import chainer
    import numpy as np
    b = chainer.cuda.to_cpu(a)

    # eigh_val, eigh_vec = eigh(a)
    # print(eigh_val.shape, eigh_vec.shape)
    # eig_val, eig_vec = np.linalg.eig(b)
    # print("\n\n\nval=\n", eig_val, "\n\n\n\n\n\n", eigh_val)
    # print("\n\n\nvec=\n", eig_vec, "\n\n\n\n\n\n", eigh_vec)

    print("CPU: ", np.linalg.det(b), np.linalg.det(b).shape)
    print("GPU: ", det_Hermitian(a), det_Hermitian(a).shape)
