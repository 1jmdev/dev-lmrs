// extern "C" is required so cudarc can load the function by its plain name.
extern "C" __global__ void vector_add(const float* a,
                                       const float* b,
                                       float* out,
                                       int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}