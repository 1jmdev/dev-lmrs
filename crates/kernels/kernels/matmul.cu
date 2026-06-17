#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

static cublasStatus_t get_cublas_handle(cublasHandle_t* handle) {
    static thread_local cublasHandle_t cached_handle = nullptr;
    if (cached_handle == nullptr) {
        const cublasStatus_t status = cublasCreate(&cached_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            return status;
        }
    }

    *handle = cached_handle;
    return CUBLAS_STATUS_SUCCESS;
}

extern "C" int fp16_matmul_nn(half* out,
                               const half* a,
                               const half* b,
                               int rows,
                               int cols,
                               int inner,
                               int n,
                               unsigned long long stream) {
    if (n < rows * cols) {
        return 1;
    }

    cublasHandle_t handle;
    cublasStatus_t status = get_cublas_handle(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return static_cast<int>(status);
    }

    status = cublasSetStream(handle, reinterpret_cast<cudaStream_t>(stream));
    if (status != CUBLAS_STATUS_SUCCESS) {
        return static_cast<int>(status);
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    status = cublasGemmEx(handle,
                          CUBLAS_OP_N,
                          CUBLAS_OP_N,
                          cols,
                          rows,
                          inner,
                          &alpha,
                          b,
                          CUDA_R_16F,
                          cols,
                          a,
                          CUDA_R_16F,
                          inner,
                          &beta,
                          out,
                          CUDA_R_16F,
                          cols,
                          CUBLAS_COMPUTE_32F,
                          CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    return static_cast<int>(status);
}

extern "C" int fp16_matmul_nt(half* out,
                               const half* a,
                               const half* b,
                               int rows,
                               int cols,
                               int inner,
                               int n,
                               unsigned long long stream) {
    if (n < rows * cols) {
        return 1;
    }

    cublasHandle_t handle;
    cublasStatus_t status = get_cublas_handle(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return static_cast<int>(status);
    }

    status = cublasSetStream(handle, reinterpret_cast<cudaStream_t>(stream));
    if (status != CUBLAS_STATUS_SUCCESS) {
        return static_cast<int>(status);
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    status = cublasGemmEx(handle,
                          CUBLAS_OP_T,
                          CUBLAS_OP_N,
                          cols,
                          rows,
                          inner,
                          &alpha,
                          b,
                          CUDA_R_16F,
                          inner,
                          a,
                          CUDA_R_16F,
                          inner,
                          &beta,
                          out,
                          CUDA_R_16F,
                          cols,
                          CUBLAS_COMPUTE_32F,
                          CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    return static_cast<int>(status);
}
