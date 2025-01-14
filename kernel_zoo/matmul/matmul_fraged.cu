#include "common.cuh"

constexpr int TILE_SZ = 64, FRAG_SIZE = 8;

template <typename T, int _TILE_SZ, int _FRAG_SIZE>
__global__ void matmul_fraged_out(
    const T* pA, const T* pB, T* pC, size_t m, size_t n, size_t k
) {
    static_assert(_TILE_SZ % _FRAG_SIZE == 0);
    size_t tI = blockIdx.x, tJ = blockIdx.y;
    size_t fI = threadIdx.x, fJ = threadIdx.y;

    __shared__ T tA[_TILE_SZ][_TILE_SZ], tB[_TILE_SZ][_TILE_SZ];
    T fA[_FRAG_SIZE], fB[_FRAG_SIZE];
    T fC[_FRAG_SIZE][_FRAG_SIZE] = {0};

    for (size_t tK = 0; tK < cdiv_d(k, _TILE_SZ); tK++) {
        // load from gmem to smem
        #pragma unroll
        for (size_t i = 0; i < _FRAG_SIZE; i++) {
            #pragma unroll
            for (size_t j = 0; j < _FRAG_SIZE; j++) {
                tA[fI * _FRAG_SIZE + i][fJ * _FRAG_SIZE + j] = ((tI * _TILE_SZ + fI * _FRAG_SIZE + i) < m && (tK * _TILE_SZ + fJ * _FRAG_SIZE + j) < k) ? pA[(tI * _TILE_SZ + fI * _FRAG_SIZE + i) * k + tK * _TILE_SZ + fJ * _FRAG_SIZE + j] : 0;
                tB[fI * _FRAG_SIZE + i][fJ * _FRAG_SIZE + j] = ((tK * _TILE_SZ + fI * _FRAG_SIZE + i) < k && (tJ * _TILE_SZ + fJ * _FRAG_SIZE + j) < n) ? pB[(tK * _TILE_SZ + fI * _FRAG_SIZE + i) * n + tJ * _TILE_SZ + fJ * _FRAG_SIZE + j] : 0;
            }
        }
        __syncthreads();
        // load from smem to register
        #pragma unroll
        for (size_t kk = 0; kk < _TILE_SZ; kk++) {
            #pragma unroll
            for (size_t x = 0; x < _FRAG_SIZE; x++) {
                fA[x] = tA[fI * _FRAG_SIZE + x][kk];
                fB[x] = tB[kk][fJ * FRAG_SIZE + x];
            }
            #pragma unroll
            for (size_t i = 0; i < _FRAG_SIZE; i++) {
                for (size_t j = 0; j < _FRAG_SIZE; j++) {
                    fC[i][j] += fA[i] * fB[j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (size_t i = 0; i < _FRAG_SIZE; i++) {
        #pragma unroll
        for (size_t j = 0; j < _FRAG_SIZE; j++) {
            if ((tI * _TILE_SZ + fI *_FRAG_SIZE + i) < m && (tJ * _TILE_SZ + fJ * _FRAG_SIZE + j) < n) {
                pC[(tI * _TILE_SZ + fI *_FRAG_SIZE + i) * n + (tJ * _TILE_SZ + fJ * _FRAG_SIZE + j)] = fC[i][j];
            }
        }
    }
}

torch::Tensor matmul_fraged(const torch::Tensor &m1, const torch::Tensor &m2) {
  int m = m1.size(0);
  int k = m1.size(1);
  int n = m2.size(1);

  TORCH_CHECK(k == m2.size(0), "matmul sizes don't match");

  auto out = torch::empty({m, n}, m1.options());
  size_t tileDim = cdiv(TILE_SZ, FRAG_SIZE);
  dim3 tShape(tileDim, tileDim);
  dim3 bShape(cdiv(m, TILE_SZ), cdiv(n, TILE_SZ));
  matmul_fraged_out<float, TILE_SZ, FRAG_SIZE><<<bShape, tShape>>>(m1.data_ptr<float>(),
                                       m2.data_ptr<float>(),
                                       out.data_ptr<float>(), m, n, k);
  return out;
}
