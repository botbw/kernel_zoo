import os

import torch

from utils import profile, compile_cuda_module

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


def test_reduce_sum_simple():
    torch.cuda.reset_max_memory_allocated()
    with open(f"{CUR_DIR}/reduce_sum_simple.cu", 'r') as f:
        cuda_source = f.read()

    ext = compile_cuda_module('reduce_sum_simple', cuda_source,
                              CUR_DIR + '/build_reduce_sum_simple')
    m1 = torch.ones(1024, device='cuda')

    assert ext.reduce_sum_simple(m1).item() == 1024

    print(profile(ext.reduce_sum_simple, m1))
    print(
        f"Max memory allocated: {torch.cuda.max_memory_allocated()/1024 ** 2} MB"
    )


def test_reduce_sum_shared_mem():
    torch.cuda.reset_max_memory_allocated()
    with open(f"{CUR_DIR}/reduce_sum_shared_mem.cu", 'r') as f:
        cuda_source = f.read()

    ext = compile_cuda_module('reduce_sum_shared_mem', cuda_source,
                              CUR_DIR + '/build_reduce_sum_shared_mem')
    m1 = torch.ones(1024, device='cuda')

    assert ext.reduce_sum_shared_mem(m1).item() == 1024

    print(profile(ext.reduce_sum_shared_mem, m1))
    print(
        f"Max memory allocated: {torch.cuda.max_memory_allocated()/1024 ** 2} MB"
    )


if __name__ == '__main__':
    test_reduce_sum_simple()
    test_reduce_sum_shared_mem()
