#include <cuda_fp16.h>

extern "C" __global__ void fp16_rope(half* out,
                                      const half* x,
                                      int seq_len,
                                      int head_dim,
                                      float theta,
                                      int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const int d = i % head_dim;
        const int pair = d & ~1;
        const int pos = (i / head_dim) % seq_len;
        const int base = i - d + pair;
        const float inv_freq = powf(theta, -static_cast<float>(pair) / static_cast<float>(head_dim));
        float s, c;
        __sincosf(static_cast<float>(pos) * inv_freq, &s, &c);
        const float x0 = __half2float(x[base]);
        const float x1 = __half2float(x[base + 1]);
        const float y = (d & 1) ? x1 * c + x0 * s : x0 * c - x1 * s;
        out[i] = __float2half_rn(y);
    }
}
