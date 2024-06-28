import os
import re

from torch.utils.cpp_extension import load_inline

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

with open(f"{CUR_DIR}/csrc/common.cuh", 'r') as f:
    COMMON_HEADER = f.read()


def get_sig(fname, src):
    res = re.findall(rf'^(.+\s+{fname}\(.*?\))\s*{{?\s*$', src, re.MULTILINE)
    return res[0] + ';' if res else None


def compile_cuda_module(func_name,
                        cuda_src,
                        build_dir,
                        extra_cuda_cflags=None):
    cpp_src = get_sig(func_name, cuda_src)

    if not os.path.exists(build_dir):
        os.makedirs(build_dir)

    ext = load_inline(
        name=func_name,
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        functions=[func_name],
        with_cuda=True,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=True,
        build_directory=build_dir,
        extra_include_paths=[f"{CUR_DIR}/csrc"],
    )

    return ext


def compile_cpp_module(func_name, cpp_src, build_dir, extra_cuda_cflags=None):
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)

    ext = load_inline(
        name=func_name,
        cpp_sources=[cpp_src],
        functions=[func_name],
        verbose=True,
        build_directory=build_dir,
        extra_include_paths=[f"{CUR_DIR}/csrc"],
    )

    return ext
