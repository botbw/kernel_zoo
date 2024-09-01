#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
// https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf
// https://github.com/tspeterkim/flash-attention-minimal/blob/main/flash.cu

#include "common.cuh"

__global__ void flash_attn_out(
    const float *q, const float *k, const float *v,
    float *l, float *m,
    float *attn_o,
    const int seql, const int head_dim, const float rooted_head_dim, const int Br, const int Bc, const int Tr, const int Tc)
{
    const int tid = threadIdx.x;
    const int batch_i = blockIdx.x, head_i = blockIdx.y;
    const int num_heads = gridDim.y;

    const int offset_qkv = batch_i * num_heads * seql * head_dim + head_i * seql * head_dim;
    const int offset_lm = batch_i * num_heads * seql + head_i * seql;

    extern __shared__ float sram[];
    const int shift = Bc * head_dim;
    float *q_tile = sram;
    float *k_tile = &sram[shift];
    float *v_tile = &sram[shift * 2];
    float *logits = &sram[shift * 3];

    for (int j = 0; j < Tc; j++)
    {
        for (int x = 0; x < head_dim; x++)
        {
            k_tile[tid * head_dim + x] = k[offset_qkv + j * shift + tid * head_dim + x];
            v_tile[tid * head_dim + x] = v[offset_qkv + j * shift + tid * head_dim + x];
        }
        __syncthreads();
        for (int i = 0; i < Tr; i++)
        {
            float m_prev = m[offset_lm + i * Br + tid];
            float l_prev = l[offset_lm + i * Br + tid];

            for (int x = 0; x < head_dim; x++)
            {
                q_tile[tid * head_dim + x] = q[offset_qkv + i * shift + tid * head_dim + x];
            }

            float m_cur = -INFINITY;
            for (int jj = 0; jj < Bc; jj++)
            {
                float s = 0;
                for (int x = 0; x < head_dim; x++)
                {
                    s += q_tile[tid * head_dim + x] * k_tile[jj * head_dim + x];
                }
                s /= rooted_head_dim;
                logits[tid * Bc + jj] = s;
                m_cur = s > m_cur ? s : m_cur;
            }
            assert(m_cur != -INFINITY);

            float l_cur = 0;
            for (int jj = 0; jj < Bc; jj++)
            {
                logits[tid * Bc + jj] = __expf(logits[tid * Bc + jj] - m_cur);
                l_cur += logits[tid * Bc + jj];
            }

            float m_new = m_cur > m_prev ? m_cur : m_prev;
            float l_new = l_prev * __expf(m_prev - m_new) + l_cur * __expf(m_cur - m_new);
            m[offset_lm + i * Br + tid] = m_new;
            l[offset_lm + i * Br + tid] = l_new;

            for (int x = 0; x < head_dim; x++)
            {
                float o_cur = 0;
                for (int jj = 0; jj < Bc; jj++)
                {
                    o_cur += logits[tid * Bc + jj] * v_tile[jj * head_dim + x];
                }
                float& o_new = attn_o[offset_qkv + i * Br * head_dim + tid * head_dim + x];
                o_new = (o_new * l_prev * __expf(m_prev - m_new) + o_cur * __expf(m_cur - m_new)) / l_new;
            }
        }
        __syncthreads();
    }
}

torch::Tensor flash_attn(torch::Tensor q, torch::Tensor k, torch::Tensor v)
{
    const int bsz = q.size(0);
    const int num_heads = q.size(1);
    const int seql = q.size(2);
    const int head_dim = q.size(3);
    const float rooted_head_dim = sqrt(head_dim);

    const int Br = 32;
    const int Bc = Br < head_dim ? Br : head_dim;
    const int Tr = cdiv(seql, Br), Tc = cdiv(seql, Bc);

    torch::Tensor attn_o = torch::zeros_like(q);
    const torch::Device dev = attn_o.device();
    torch::Tensor m = torch::full({bsz, num_heads, seql}, -INFINITY, torch::TensorOptions().device(dev));
    torch::Tensor l = torch::zeros_like(m);

    const int local_qkv_size = Bc * head_dim * 3 * sizeof(float);
    const int local_logits_size = Br * Bc * sizeof(float);
    const int sram_size = local_qkv_size + local_logits_size;

    dim3 grid(bsz, num_heads);
    dim3 blk(Bc);

    flash_attn_out<<<grid, blk, sram_size>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        l.data_ptr<float>(), m.data_ptr<float>(),
        attn_o.data_ptr<float>(),
        seql, head_dim, rooted_head_dim, Br, Bc, Tr, Tc);

    return attn_o;
}