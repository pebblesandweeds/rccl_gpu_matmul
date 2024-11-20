#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rccl/rccl.h>
#include "../include/utils.h"

#define N 32768
#define NUM_RUNS 25

// Helper function to check NCCL status
#define CHECK_NCCL(cmd) do {                         \
    ncclResult_t e = (cmd);                         \
    if (e != ncclSuccess) {                         \
        printf("NCCL error %s:%d '%s'\n",           \
            __FILE__,__LINE__,ncclGetErrorString(e));\
        exit(EXIT_FAILURE);                         \
    }                                               \
} while(0)

int main(int argc, char *argv[]) {
    int num_gpus;
    CHECK_HIP(hipGetDeviceCount(&num_gpus));

    // Allocate and initialize host matrices on GPU 0
    float *h_A = NULL, *h_B = NULL, *h_C = NULL;
    size_t full_size = N * N * sizeof(float);
    size_t chunk_size = N / num_gpus;
    size_t chunk_bytes = chunk_size * N * sizeof(float);

    h_A = (float*)malloc(full_size);
    h_B = (float*)malloc(full_size);
    h_C = (float*)malloc(full_size);

    // Initialize matrices with random data
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    // Arrays to store device pointers for each GPU
    float **d_A_chunks = (float**)malloc(num_gpus * sizeof(float*));
    float **d_B = (float**)malloc(num_gpus * sizeof(float*));
    float **d_C_chunks = (float**)malloc(num_gpus * sizeof(float*));

    // NCCL communicators
    ncclComm_t* comms = (ncclComm_t*)malloc(sizeof(ncclComm_t) * num_gpus);
    hipStream_t* streams = (hipStream_t*)malloc(sizeof(hipStream_t) * num_gpus);

    // Initialize NCCL communicators
    int devList[num_gpus];
    for (int i = 0; i < num_gpus; i++) {
        devList[i] = i;
    }
    CHECK_NCCL(ncclCommInitAll(comms, num_gpus, devList));

    // Allocate memory and create streams on each GPU
    for (int i = 0; i < num_gpus; i++) {
        CHECK_HIP(hipSetDevice(i));

        // Create stream for this GPU
        CHECK_HIP(hipStreamCreate(&streams[i]));

        // Allocate memory on this GPU
        CHECK_HIP(hipMalloc(&d_A_chunks[i], chunk_bytes));
        CHECK_HIP(hipMalloc(&d_B[i], full_size));
        CHECK_HIP(hipMalloc(&d_C_chunks[i], chunk_bytes));

        // Copy chunk of A and full B to each GPU
        CHECK_HIP(hipMemcpyAsync(d_A_chunks[i],
                                h_A + (i * chunk_size * N),
                                chunk_bytes,
                                hipMemcpyHostToDevice,
                                streams[i]));

        CHECK_HIP(hipMemcpyAsync(d_B[i],
                                h_B,
                                full_size,
                                hipMemcpyHostToDevice,
                                streams[i]));
    }

    // Create rocBLAS handles for each GPU
    rocblas_handle* handles = (rocblas_handle*)malloc(num_gpus * sizeof(rocblas_handle));
    for (int i = 0; i < num_gpus; i++) {
        CHECK_HIP(hipSetDevice(i));
        CHECK_ROCBLAS(rocblas_create_handle(&handles[i]));
        CHECK_ROCBLAS(rocblas_set_stream(handles[i], streams[i]));
    }

    // Broadcast matrix B to all GPUs using NCCL
    for (int i = 0; i < num_gpus; i++) {
        CHECK_HIP(hipSetDevice(i));
        CHECK_NCCL(ncclBroadcast(d_B[i], d_B[i], N * N, ncclFloat, 0,
                                comms[i], streams[i]));
    }

    // Perform matrix multiplication on each GPU
    const float alpha = 1.0f;
    const float beta = 0.0f;

    for (int run = 0; run < NUM_RUNS; run++) {
        hipEvent_t starts[num_gpus], stops[num_gpus];

        for (int i = 0; i < num_gpus; i++) {
            CHECK_HIP(hipSetDevice(i));
            CHECK_HIP(hipEventCreate(&starts[i]));
            CHECK_HIP(hipEventCreate(&stops[i]));
            CHECK_HIP(hipEventRecord(starts[i], streams[i]));

            CHECK_ROCBLAS(rocblas_sgemm(handles[i],
                                       rocblas_operation_none,
                                       rocblas_operation_none,
                                       N, chunk_size, N,
                                       &alpha,
                                       d_B[i], N,
                                       d_A_chunks[i], N,
                                       &beta,
                                       d_C_chunks[i], N));

            CHECK_HIP(hipEventRecord(stops[i], streams[i]));
        }

        // Wait for computations and measure time
        for (int i = 0; i < num_gpus; i++) {
            CHECK_HIP(hipSetDevice(i));
            CHECK_HIP(hipEventSynchronize(stops[i]));

            float compute_time;
            CHECK_HIP(hipEventElapsedTime(&compute_time, starts[i], stops[i]));
            double chunk_flops = 2.0 * chunk_size * N * N;
            double tflops = (chunk_flops / (compute_time / 1000.0)) / 1e12;

            printf("GPU %d, Run %d: Time: %.2f ms, Performance: %.2f TFLOPS\n",
                   i, run+1, compute_time, tflops);

            CHECK_HIP(hipEventDestroy(starts[i]));
            CHECK_HIP(hipEventDestroy(stops[i]));
        }
    }

    // Gather results using NCCL AllGather
    for (int i = 0; i < num_gpus; i++) {
        CHECK_HIP(hipSetDevice(i));
        CHECK_NCCL(ncclAllGather(d_C_chunks[i], d_C_chunks[i], chunk_size * N,
                                ncclFloat, comms[i], streams[i]));
    }

    // Copy final results back to host
    CHECK_HIP(hipMemcpy(h_C, d_C_chunks[0], full_size, hipMemcpyDeviceToHost));

    // Cleanup
    for (int i = 0; i < num_gpus; i++) {
        CHECK_HIP(hipSetDevice(i));
        CHECK_ROCBLAS(rocblas_destroy_handle(handles[i]));
        CHECK_HIP(hipStreamDestroy(streams[i]));
        CHECK_HIP(hipFree(d_A_chunks[i]));
        CHECK_HIP(hipFree(d_B[i]));
        CHECK_HIP(hipFree(d_C_chunks[i]));
    }

    // Destroy NCCL communicators
    for (int i = 0; i < num_gpus; i++) {
        CHECK_NCCL(ncclCommDestroy(comms[i]));
    }

    free(comms);
    free(streams);
    free(handles);
    free(d_A_chunks);
    free(d_B);
    free(d_C_chunks);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
