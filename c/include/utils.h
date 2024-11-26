#ifndef UTILS_H
#define UTILS_H
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rccl/rccl.h>

#define CHECK_HIP(stmt) do {                                 \
    hipError_t err = stmt;                                   \
    if (err != hipSuccess) {                                 \
        printf("HIP error: %s\n", hipGetErrorString(err));   \
        exit(1);                                             \
    }                                                        \
} while(0)

#define CHECK_ROCBLAS(stmt) do {                             \
    rocblas_status status = stmt;                            \
    if (status != rocblas_status_success) {                  \
        printf("rocBLAS error: %d\n", status);               \
        exit(1);                                             \
    }                                                        \
} while(0)

#define CHECK_NCCL(stmt) do {                                \
    ncclResult_t status = stmt;                              \
    if (status != ncclSuccess) {                             \
        printf("NCCL error: %s\n", ncclGetErrorString(status)); \
        exit(1);                                             \
    }                                                        \
} while(0)

void print_gpu_info();
void print_precision();
const char* get_precision_string(size_t size);
void cleanup_resources(rocblas_handle* handles, float** d_A_chunks, float** d_B,
                      float** d_C_chunks, float** d_C_final, float* h_A,
                      float* h_B, float* h_C, int num_gpus);
#endif // UTILS_H
