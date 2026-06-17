#include <cuda_fp16.h>

extern "C" __global__ void fp16_add(half* out, const half* a, const half* b, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        out[i] = __hadd(a[i], b[i]);
    }
}

extern "C" __global__ void fp16_sub(half* out, const half* a, const half* b, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        out[i] = __hsub(a[i], b[i]);
    }
}

extern "C" __global__ void fp16_mul(half* out, const half* a, const half* b, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        out[i] = __hmul(a[i], b[i]);
    }
}

extern "C" __global__ void fp16_div(half* out, const half* a, const half* b, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float v = __half2float(a[i]) / __half2float(b[i]);
        out[i] = __float2half_rn(v);
    }
}

extern "C" __global__ void fp16_add_scalar(half* out, const half* x, float value, int n) {
    const half v = __float2half_rn(value);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        out[i] = __hadd(x[i], v);
    }
}

extern "C" __global__ void fp16_mul_scalar(half* out, const half* x, float value, int n) {
    const half v = __float2half_rn(value);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        out[i] = __hmul(x[i], v);
    }
}

extern "C" __global__ void fp16_relu(half* out, const half* x, int n) {
    const half zero = __float2half_rn(0.0f);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const half v = x[i];
        out[i] = __hgt(v, zero) ? v : zero;
    }
}

extern "C" __global__ void fp16_silu(half* out, const half* x, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const float v = __half2float(x[i]);
        out[i] = __float2half_rn(v / (1.0f + __expf(-v)));
    }
}

extern "C" __global__ void fp16_gelu(half* out, const half* x, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const float v = __half2float(x[i]);
        const float y = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
        out[i] = __float2half_rn(y);
    }
}

extern "C" __global__ void fp16_exp(half* out, const half* x, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        out[i] = __float2half_rn(__expf(__half2float(x[i])));
    }
}

extern "C" __global__ void fp16_tanh(half* out, const half* x, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        out[i] = __float2half_rn(tanhf(__half2float(x[i])));
    }
}
