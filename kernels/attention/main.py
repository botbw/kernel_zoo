import os
import torch

from utils import profile, compile_cuda_module
from torch.testing import assert_allclose
CUR_DIR = os.path.dirname(os.path.abspath(__file__))

BSZ = 16
NUM_HEADS=8
HEAD_DIM=64
SEQ_LEN=512

def torch_attn(q, k, v):
    assert len(q.shape) == 4
    bsz, num_heads, seq_len, head_dim = q.shape
    attn = torch.matmul(q, k.transpose(-1, -2)) / head_dim ** 0.5
    attn = torch.nn.functional.softmax(attn, dim=-1)
    return torch.matmul(attn, v)

def test_flash_attn_v1():
    torch.cuda.reset_max_memory_allocated()
    with open(f"{CUR_DIR}/flash_attn.cu", 'r') as f:
        cuda_source = f.read()

    flash_attn = compile_cuda_module('flash_attn', cuda_source,
                              CUR_DIR + '/build_flash_attn')
    
    q = torch.randn(BSZ, NUM_HEADS, SEQ_LEN, HEAD_DIM, device='cuda')
    k = torch.randn(BSZ, NUM_HEADS, SEQ_LEN, HEAD_DIM, device='cuda')
    v = torch.randn(BSZ, NUM_HEADS, SEQ_LEN, HEAD_DIM, device='cuda')

    out_torch = torch_attn(q, k, v)
    out_flash = flash_attn.flash_attn(q, k, v)

    # print(out_flash)
    assert_allclose(out_torch, out_flash, atol=1e-4, rtol=1e-3)

    torch.cuda.reset_max_memory_allocated()
    print(profile(flash_attn.flash_attn, q, k, v))
    print(
        f"Max memory allocated: {torch.cuda.max_memory_allocated()/1024 ** 2} MB"
    )

    torch.cuda.reset_max_memory_allocated()
    print(profile(torch_attn, q, k, v))
    print(
        f"Max memory allocated: {torch.cuda.max_memory_allocated()/1024 ** 2} MB"
    )

if __name__ == '__main__':
    test_flash_attn_v1()