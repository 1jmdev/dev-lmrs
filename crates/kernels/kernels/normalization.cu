#include <cuda_fp16.h>

static __device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

extern "C" __global__ void fp16_rms_norm(half* out,
                                          const half* x,
                                          const half* weight,
                                          int width,
                                          float eps,
                                          int n) {
    __shared__ float smem[64];
    const int rows = n / width;
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {

        float ss = 0.0f;
        for (int col = threadIdx.x; col < width; col += blockDim.x) {
            const float v = __half2float(x[row * width + col]);
            ss += v * v;
        }
        ss = warp_sum(ss);
        if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = ss;
        __syncthreads();
        ss = threadIdx.x < (blockDim.x >> 5) ? smem[threadIdx.x] : 0.0f;
        if (threadIdx.x < 32) ss = warp_sum(ss);
        if (threadIdx.x == 0) smem[0] = rsqrtf(ss / width + eps);
        __syncthreads();
        const float inv = smem[0];

        for (int col = threadIdx.x; col < width; col += blockDim.x) {
            const float v = __half2float(x[row * width + col]);
            const float w = __half2float(weight[col]);
            out[row * width + col] = __float2half_rn(v * inv * w);
        }
        __syncthreads();
    }
}

extern "C" __global__ void fp16_layer_norm(half* out,
                                            const half* x,
                                            const half* weight,
                                            const half* bias,
                                            int width,
                                            float eps,
                                            int n) {
    __shared__ float smem[64];
    const int rows = n / width;
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {

        float sum = 0.0f;
        float sum2 = 0.0f;
        for (int col = threadIdx.x; col < width; col += blockDim.x) {
            const float v = __half2float(x[row * width + col]);
            sum += v;
            sum2 += v * v;
        }
        sum = warp_sum(sum);
        sum2 = warp_sum(sum2);
        if ((threadIdx.x & 31) == 0) {
            smem[threadIdx.x >> 5] = sum;
            smem[(blockDim.x >> 5) + (threadIdx.x >> 5)] = sum2;
        }
        __syncthreads();

        sum = threadIdx.x < (blockDim.x >> 5) ? smem[threadIdx.x] : 0.0f;
        sum2 = threadIdx.x < (blockDim.x >> 5) ? smem[(blockDim.x >> 5) + threadIdx.x] : 0.0f;
        if (threadIdx.x < 32) {
            sum = warp_sum(sum);
            sum2 = warp_sum(sum2);
        }
        if (threadIdx.x == 0) {
            const float mean = sum / width;
            smem[0] = mean;
            smem[1] = rsqrtf(fmaxf(sum2 / width - mean * mean, 0.0f) + eps);
        }
        __syncthreads();

        const float mean = smem[0];
        const float inv = smem[1];
        for (int col = threadIdx.x; col < width; col += blockDim.x) {
            const float v = __half2float(x[row * width + col]);
            const float w = __half2float(weight[col]);
            const float b = __half2float(bias[col]);
            out[row * width + col] = __float2half_rn((v - mean) * inv * w + b);
        }
        __syncthreads();
    }
}
