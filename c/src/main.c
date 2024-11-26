#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rccl/rccl.h>
#include "../include/utils.h"
#include "../include/spot_check.h"
#include "../include/matrix_operations.h"
#include "../include/rccl_utils.h"

#define N 32768
#define NUM_RUNS 25

int main(int argc, char *argv[]) {
    print_gpu_info();
    print_precision();
    printf("Matrix size: %d x %d, using %s precision\n", N, N, get_precision_string(sizeof(float)));
    printf("\n");

    int num_gpus;
    CHECK_HIP(hipGetDeviceCount(&num_gpus));
    printf("Running on %d GPUs\n\n", num_gpus);

    float *h_A = NULL, *h_B = NULL, *h_C = NULL;
    size_t full_size = N * N * sizeof(float);
    size_t chunk_size = N / num_gpus;
    size_t chunk_bytes = chunk_size * N * sizeof(float);

    h_A = (float*)malloc(full_size);
    h_B = (float*)malloc(full_size);
    h_C = (float*)malloc(full_size);
    initialize_matrices(h_A, h_B, N);

    float **d_A_chunks = (float**)malloc(num_gpus * sizeof(float*));
    float **d_B = (float**)malloc(num_gpus * sizeof(float*));
    float **d_C_chunks = (float**)malloc(num_gpus * sizeof(float*));
    float **d_C_final = (float**)malloc(num_gpus * sizeof(float*));

    // Initialize RCCL context
    RCCLContext* rccl_ctx = rccl_init(num_gpus);

    for (int i = 0; i < num_gpus; i++) {
        CHECK_HIP(hipSetDevice(i));
        CHECK_HIP(hipMalloc(&d_A_chunks[i], chunk_bytes));
        CHECK_HIP(hipMalloc(&d_B[i], full_size));
        CHECK_HIP(hipMalloc(&d_C_chunks[i], chunk_bytes));
        CHECK_HIP(hipMalloc(&d_C_final[i], full_size));
        CHECK_HIP(hipMemcpyAsync(d_A_chunks[i],
                                h_A + (i * chunk_size * N),
                                chunk_bytes,
                                hipMemcpyHostToDevice,
                                rccl_ctx->streams[i]));
        CHECK_HIP(hipMemcpyAsync(d_B[i],
                                h_B,
                                full_size,
                                hipMemcpyHostToDevice,
                                rccl_ctx->streams[i]));
    }

    rocblas_handle* handles = (rocblas_handle*)malloc(num_gpus * sizeof(rocblas_handle));
    for (int i = 0; i < num_gpus; i++) {
        CHECK_HIP(hipSetDevice(i));
        CHECK_ROCBLAS(rocblas_create_handle(&handles[i]));
        CHECK_ROCBLAS(rocblas_set_stream(handles[i], rccl_ctx->streams[i]));
    }

    // Broadcast matrix B to all GPUs
    printf("Broadcasting matrix B to all GPUs\n");
    rccl_broadcast_matrix(rccl_ctx, d_B, N * N);
    rccl_sync_and_check(rccl_ctx);

    // Perform matrix multiplication
    perform_matrix_multiplication(handles, d_A_chunks, d_B, d_C_chunks, N, chunk_size, num_gpus, rccl_ctx->streams, NUM_RUNS);

    // Initialize final result arrays
    for (int i = 0; i < num_gpus; i++) {
        CHECK_HIP(hipSetDevice(i));
        CHECK_HIP(hipMemsetAsync(d_C_final[i], 0, full_size, rccl_ctx->streams[i]));
    }

    // Gather results from all GPUs
    printf("Starting AllGather\n");
    rccl_gather_matrix_chunks(rccl_ctx, d_C_chunks, d_C_final, chunk_size * N);
    printf("Waiting for RCCL operations\n");
    rccl_sync_and_check(rccl_ctx);

    printf("Copying results back to host\n");
    CHECK_HIP(hipSetDevice(0));
    CHECK_HIP(hipMemcpy(h_C, d_C_final[0], full_size, hipMemcpyDeviceToHost));

    printf("\nStarting spot check validation...\n");
    spot_check(h_A, h_B, h_C, N);
    printf("Spot check complete\n\n");

    // Cleanup RCCL context and resources
    rccl_cleanup(rccl_ctx);
    cleanup_resources(handles, d_A_chunks, d_B, d_C_chunks, d_C_final, h_A, h_B, h_C, num_gpus);

    return 0;
}
