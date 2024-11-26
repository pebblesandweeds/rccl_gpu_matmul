#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#include <rocblas/rocblas.h>

void initialize_matrices(float *A, float *B, int n);
void perform_matrix_multiplication(
    rocblas_handle* handles,
    float** d_A_chunks,
    float** d_B,
    float** d_C_chunks,
    int N,
    int chunk_size,
    int num_gpus,
    hipStream_t* streams,
    int NUM_RUNS);

#endif // MATRIX_OPERATIONS_H
