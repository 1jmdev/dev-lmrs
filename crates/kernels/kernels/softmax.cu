#include <cuda_fp16.h>

#define LMRS_NEG_INF -3.4028234663852886e38f

static __device__ __forceinline__ float warp_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    }
    return v;
}

static __device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

extern "C" __global__ void fp16_softmax_rows(half* out,
                                              const half* x,
                                              int width,
                                              float scale,
                                              int n) {
    __shared__ float smem[32];
    const int rows = n / width;
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {

        float max_v = LMRS_NEG_INF;
        for (int col = threadIdx.x; col < width; col += blockDim.x) {
            max_v = fmaxf(max_v, __half2float(x[row * width + col]) * scale);
        }
        max_v = warp_max(max_v);
        if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = max_v;
        __syncthreads();
        max_v = threadIdx.x < (blockDim.x >> 5) ? smem[threadIdx.x] : LMRS_NEG_INF;
        if (threadIdx.x < 32) max_v = warp_max(max_v);
        if (threadIdx.x == 0) smem[0] = max_v;
        __syncthreads();
        max_v = smem[0];

        float sum = 0.0f;
        for (int col = threadIdx.x; col < width; col += blockDim.x) {
            sum += __expf(__half2float(x[row * width + col]) * scale - max_v);
        }
        sum = warp_sum(sum);
        if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = sum;
        __syncthreads();
        sum = threadIdx.x < (blockDim.x >> 5) ? smem[threadIdx.x] : 0.0f;
        if (threadIdx.x < 32) sum = warp_sum(sum);
        if (threadIdx.x == 0) smem[0] = 1.0f / sum;
        __syncthreads();
        const float inv_sum = smem[0];

        for (int col = threadIdx.x; col < width; col += blockDim.x) {
            const float y = __expf(__half2float(x[row * width + col]) * scale - max_v) * inv_sum;
            out[row * width + col] = __float2half_rn(y);
        }
        __syncthreads();
    }
}

extern "C" __global__ void fp16_causal_mask(half* out,
                                             const half* x,
                                             int query_len,
                                             int key_len,
                                             int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const int q = (i / key_len) % query_len;
        const int k = i % key_len;
        const float v = k <= q ? __half2float(x[i]) : -65504.0f;
        out[i] = __float2half_rn(v);
    }
}
