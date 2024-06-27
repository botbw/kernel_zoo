import os

import torch
from torch.testing import assert_allclose

from utils import profile, compile_module

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

MAT_SIZE = 2048


def test_matmul_simple():
    with open(f"{CUR_DIR}/matmul_simple.cu", 'r') as f:
        cuda_source = f.read()

    ext = compile_module('matmul_simple', cuda_source,
                         CUR_DIR + '/build_matmul_simple')

    m1 = torch.randn(MAT_SIZE, MAT_SIZE, device='cuda')
    m2 = torch.randn(MAT_SIZE, MAT_SIZE, device='cuda')

    assert_allclose(ext.matmul_simple(m1, m2), m1 @ m2)

    print(profile(ext.matmul_simple, m1, m2))


def test_matmul_tiled():
    with open(f"{CUR_DIR}/matmul_tiled.cu", 'r') as f:
        cuda_source = f.read()

    ext = compile_module('matmul_tiled', cuda_source,
                         CUR_DIR + '/build_matmul_tiled')

    m1 = torch.randn(MAT_SIZE, MAT_SIZE, device='cuda')
    m2 = torch.randn(MAT_SIZE, MAT_SIZE, device='cuda')

    assert_allclose(ext.matmul_tiled(m1, m2), m1 @ m2)

    print(profile(ext.matmul_tiled, m1, m2))


if __name__ == '__main__':
    test_matmul_simple()
    test_matmul_tiled()
