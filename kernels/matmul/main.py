import os
import sys
import importlib

import torch
from torch.testing import assert_allclose

from utils import profile, compile_module

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

MAT_SIZE = 2048


def test_torch_native():
    torch.cuda.reset_max_memory_allocated()

    m1 = torch.randn(MAT_SIZE, MAT_SIZE, device='cuda')
    m2 = torch.randn(MAT_SIZE, MAT_SIZE, device='cuda')

    print(profile(torch.matmul, m1, m2))
    print(
        f"Max memory allocated: {torch.cuda.max_memory_allocated()/1024 ** 2} MB"
    )


def test_matmul_simple():
    with open(f"{CUR_DIR}/matmul_simple.cu", 'r') as f:
        cuda_source = f.read()

    ext = compile_module('matmul_simple', cuda_source,
                         CUR_DIR + '/build_matmul_simple')

    m1 = torch.randn(MAT_SIZE, MAT_SIZE, device='cuda')
    m2 = torch.randn(MAT_SIZE, MAT_SIZE, device='cuda')

    assert_allclose(ext.matmul_simple(m1, m2), m1 @ m2)

    print(profile(ext.matmul_simple, m1, m2))
    print(
        f"Max memory allocated: {torch.cuda.max_memory_allocated()/1024 ** 2} MB"
    )


def test_matmul_tiled():
    torch.cuda.reset_max_memory_allocated()
    with open(f"{CUR_DIR}/matmul_tiled.cu", 'r') as f:
        cuda_source = f.read()

    ext = compile_module('matmul_tiled', cuda_source,
                         CUR_DIR + '/build_matmul_tiled')

    m1 = torch.randn(MAT_SIZE, MAT_SIZE, device='cuda')
    m2 = torch.randn(MAT_SIZE, MAT_SIZE, device='cuda')

    assert_allclose(ext.matmul_tiled(m1, m2), m1 @ m2)

    print(profile(ext.matmul_tiled, m1, m2))
    print(
        f"Max memory allocated: {torch.cuda.max_memory_allocated()/1024 ** 2} MB"
    )


def test_matmul_tiled_numba():
    torch.cuda.reset_max_memory_allocated()
    spec = importlib.util.spec_from_file_location(
        'ext', f"{CUR_DIR}/matmul_tiled_numba.py")
    ext = importlib.util.module_from_spec(spec)
    sys.modules['ext'] = ext
    spec.loader.exec_module(ext)

    m1 = torch.randn(MAT_SIZE, MAT_SIZE, device='cuda')
    m2 = torch.randn(MAT_SIZE, MAT_SIZE, device='cuda')

    assert_allclose(ext.matmul_2d_numba(m1, m2, 32), m1 @ m2)

    print(profile(ext.matmul_2d_numba, m1, m2, 32))
    print(
        f"Max memory allocated: {torch.cuda.max_memory_allocated()/1024 ** 2} MB"
    )


if __name__ == '__main__':
    test_torch_native()
    test_matmul_simple()
    test_matmul_tiled()
    test_matmul_tiled_numba()
