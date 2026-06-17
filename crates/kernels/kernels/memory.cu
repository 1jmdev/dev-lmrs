#include <cuda_fp16.h>

extern "C" __global__ void fp16_fill(half* out, float value, int n) {
    const half v = __float2half_rn(value);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        out[i] = v;
    }
}

extern "C" __global__ void fp16_copy(half* out, const half* x, int n) {
    const int4* src4 = reinterpret_cast<const int4*>(x);
    int4* dst4 = reinterpret_cast<int4*>(out);
    int vec_n = n >> 3;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_n; i += blockDim.x * gridDim.x) {
        dst4[i] = src4[i];
    }
    for (int i = (vec_n << 3) + blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        out[i] = x[i];
    }
}

extern "C" __global__ void fp32_fill(float* out, float value, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        out[i] = value;
    }
}
