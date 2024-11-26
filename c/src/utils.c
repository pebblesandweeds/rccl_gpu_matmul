#include <stdio.h>
#include <stdlib.h>
#include "../include/utils.h"

void print_gpu_info() {
    hipDevice_t device;
    hipDeviceProp_t props;
    CHECK_HIP(hipGetDevice(&device));
    CHECK_HIP(hipGetDeviceProperties(&props, device));
    printf("GPU: %s\n", props.name);
    printf("Total GPU memory: %zu MB\n", props.totalGlobalMem / (1024 * 1024));
    printf("GPU clock rate: %d MHz\n", props.clockRate / 1000);
}

void print_precision() {
    printf("Matrix Element Precision: %s\n", get_precision_string(sizeof(float)));
    printf("rocBLAS Function: rocblas_sgemm (Single Precision)\n");
}

const char* get_precision_string(size_t size) {
    switch(size) {
        case sizeof(float):
            return "Single Precision (32-bit)";
        case sizeof(double):
            return "Double Precision (64-bit)";
        default:
            return "Unknown Precision";
    }
}

void cleanup_resources(rocblas_handle* handles, float** d_A_chunks, float** d_B,
                      float** d_C_chunks, float** d_C_final, float* h_A,
                      float* h_B, float* h_C, int num_gpus) {
    // Cleanup GPU resources
    for (int i = 0; i < num_gpus; i++) {
        CHECK_HIP(hipSetDevice(i));
        CHECK_ROCBLAS(rocblas_destroy_handle(handles[i]));
        CHECK_HIP(hipFree(d_A_chunks[i]));
        CHECK_HIP(hipFree(d_B[i]));
        CHECK_HIP(hipFree(d_C_chunks[i]));
        CHECK_HIP(hipFree(d_C_final[i]));
    }

    // Free CPU resources
    free(handles);
    free(d_A_chunks);
    free(d_B);
    free(d_C_chunks);
    free(d_C_final);
    free(h_A);
    free(h_B);
    free(h_C);
}
