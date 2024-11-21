#include <stdlib.h>
#include <stdio.h>
#include "../include/matrix_operations.h"
#include "../include/utils.h"

static float random_float() {
    return (float)rand() / ((float)RAND_MAX + 1.0f) * 2.0f - 1.0f;
}

void initialize_matrices(float *A, float *B, int n) {
    for (int i = 0; i < n * n; i++) {
        A[i] = random_float();
        B[i] = random_float();
    }
}

void transpose_matrix(const float *src, float *dst, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            dst[j * n + i] = src[i * n + j];
        }
    }
}

void perform_matrix_multiplication(
    rocblas_handle* handles,
    float** d_A_chunks,
    float** d_B,
    float** d_C_chunks,
    int N,
    int chunk_size,
    int num_gpus,
    hipStream_t* streams,
    int NUM_RUNS) {

    const float alpha = 1.0f;
    const float beta = 0.0f;

    printf("Starting matrix multiplication runs...\n");
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
        if (run < NUM_RUNS - 1) printf("\n");
    }

    // Sync all compute
    for (int i = 0; i < num_gpus; i++) {
        CHECK_HIP(hipSetDevice(i));
        CHECK_HIP(hipStreamSynchronize(streams[i]));
        CHECK_HIP(hipDeviceSynchronize());
    }
}
