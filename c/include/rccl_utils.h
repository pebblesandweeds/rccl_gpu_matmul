#ifndef RCCL_UTILS_H
#define RCCL_UTILS_H

#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include "utils.h"  // Include existing utils for CHECK macros

// Structure to hold RCCL communication context
typedef struct {
    ncclComm_t* comms;
    hipStream_t* streams;
    int num_gpus;
    int* device_list;
} RCCLContext;

// Initialize RCCL context and create communicators
RCCLContext* rccl_init(int num_gpus);

// Perform broadcast operation across GPUs
void rccl_broadcast_matrix(RCCLContext* ctx, float** send_data, size_t elements);

// Perform all-gather operation for matrix chunks
void rccl_gather_matrix_chunks(RCCLContext* ctx, float** chunks, float** result,
                             size_t chunk_elements);

// Synchronize all GPUs and check for errors
void rccl_sync_and_check(RCCLContext* ctx);

// Clean up RCCL resources
void rccl_cleanup(RCCLContext* ctx);

#endif // RCCL_UTILS_H
