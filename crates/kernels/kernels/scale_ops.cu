// Two kernels in one file -> two methods generated, one module load.
extern "C" __global__ void scale(float* data, float factor, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] *= factor;
    }
}

extern "C" __global__ void fill(float* data, float value, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = value;
    }
}