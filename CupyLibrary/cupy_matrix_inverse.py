#! /usr/bin/env python3
# coding;utf-8

import numpy
import cupy
from cupy import cuda

shape_array = 0

def _as_batch_mat(x):
    global shape_array
    shape_array = x.shape
    return x.reshape(-1, shape_array[-2], shape_array[-1])


def _mat_ptrs(a):
    if len(a) == 1:
        return cupy.full((1,), a.data.ptr, dtype=numpy.uintp)
    else:
        stride = a.strides[0]
        ptr = a.data.ptr
        out = cupy.arange(ptr, ptr + stride * len(a), stride, dtype=numpy.uintp)
        return out


def _get_ld(a):
    strides = a.strides[-2:]
    trans = numpy.argmin(strides)
    return trans, int(max(a.shape[trans - 2], max(strides) // a.itemsize))


def inv_gpu_batch(b):
    # We do a batched LU decomposition on the GPU to compute the inverse
    # Change the shape of the array to be size=1 minibatch if necessary
    # Also copy the matrix as the elments will be modified in-place
    a = _as_batch_mat(b).copy()
    n = a.shape[1]
    n_matrices = len(a)
    # Pivot array
    p = cupy.empty((n, n_matrices), dtype=numpy.int32)
    # Output array
    c = cupy.empty_like(a)
    # These arrays hold information on the execution success
    # or if the matrix was singular
    info = cupy.empty(n_matrices, dtype=numpy.int32)
    ap = _mat_ptrs(a)
    cp = _mat_ptrs(c)
    _, lda = _get_ld(a)
    _, ldc = _get_ld(c)
    handle = cuda.Device().cublas_handle
    if (b.dtype.char == "f"):
        cuda.cublas.sgetrfBatched(handle, n, ap.data.ptr, lda, p.data.ptr, info.data.ptr, n_matrices)
        cuda.cublas.sgetriBatched(handle, n, ap.data.ptr, lda, p.data.ptr, cp.data.ptr, ldc,info.data.ptr, n_matrices) 
    if (b.dtype.char == "d"):
        cuda.cublas.dgetrfBatched(handle, n, ap.data.ptr, lda, p.data.ptr, info.data.ptr, n_matrices)
        cuda.cublas.dgetriBatched(handle, n, ap.data.ptr, lda, p.data.ptr, cp.data.ptr, ldc,info.data.ptr, n_matrices) 
    elif (b.dtype.char == "F"):
        cuda.cublas.cgetrfBatched(handle, n, ap.data.ptr, lda, p.data.ptr, info.data.ptr, n_matrices)
        cuda.cublas.cgetriBatched(handle, n, ap.data.ptr, lda, p.data.ptr, cp.data.ptr, ldc,info.data.ptr, n_matrices)
    elif (b.dtype.char == "D"):
        cuda.cublas.zgetrfBatched(handle, n, ap.data.ptr, lda, p.data.ptr, info.data.ptr, n_matrices)
        cuda.cublas.zgetriBatched(handle, n, ap.data.ptr, lda, p.data.ptr, cp.data.ptr, ldc,info.data.ptr, n_matrices)

    if b.ndim > 3:
        return c.reshape(shape_array)
    if b.ndim == 2:
        return c[0]
    else:
        return c


if __name__ == "__main__":
    size = 4
    a = cupy.random.rand(size, size) + cupy.random.rand(size, size) * 1j
    inv_a = inv_gpu_batch(a)
    if (a @ inv_a - cupy.eye(size)).sum().real < 1e-4:
        print("cupy inverse matrix calculation work correctly")
    else:
        print("Error --- cupy inverse matrix calculation does not work --- \n Please update cupy version")