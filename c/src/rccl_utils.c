#include "rccl_utils.h"
#include <stdlib.h>
#include <stdio.h>

RCCLContext* rccl_init(int num_gpus) {
    RCCLContext* ctx = (RCCLContext*)malloc(sizeof(RCCLContext));
    ctx->num_gpus = num_gpus;
    ctx->device_list = (int*)malloc(sizeof(int) * num_gpus);
    ctx->comms = (ncclComm_t*)malloc(sizeof(ncclComm_t) * num_gpus);
    ctx->streams = (hipStream_t*)malloc(sizeof(hipStream_t) * num_gpus);

    // Initialize device list
    for (int i = 0; i < num_gpus; i++) {
        ctx->device_list[i] = i;
    }

    // Initialize RCCL communicators
    CHECK_NCCL(ncclCommInitAll(ctx->comms, num_gpus, ctx->device_list));

    // Create streams for each GPU
    for (int i = 0; i < num_gpus; i++) {
        CHECK_HIP(hipSetDevice(i));
        CHECK_HIP(hipStreamCreate(&ctx->streams[i]));
    }

    return ctx;
}

void rccl_broadcast_matrix(RCCLContext* ctx, float** send_data, size_t elements) {
    CHECK_NCCL(ncclGroupStart());
    for (int i = 0; i < ctx->num_gpus; i++) {
        CHECK_HIP(hipSetDevice(i));
        CHECK_NCCL(ncclBroadcast(send_data[i], send_data[i], elements,
                                ncclFloat, 0, ctx->comms[i], ctx->streams[i]));
    }
    CHECK_NCCL(ncclGroupEnd());
}

void rccl_gather_matrix_chunks(RCCLContext* ctx, float** chunks, float** result,
                             size_t chunk_elements) {
    CHECK_NCCL(ncclGroupStart());
    for (int i = 0; i < ctx->num_gpus; i++) {
        CHECK_HIP(hipSetDevice(i));
        CHECK_NCCL(ncclAllGather(chunks[i], result[i], chunk_elements,
                                ncclFloat, ctx->comms[i], ctx->streams[i]));
    }
    CHECK_NCCL(ncclGroupEnd());
}

void rccl_sync_and_check(RCCLContext* ctx) {
    for (int i = 0; i < ctx->num_gpus; i++) {
        CHECK_HIP(hipSetDevice(i));
        CHECK_HIP(hipStreamSynchronize(ctx->streams[i]));
        CHECK_HIP(hipDeviceSynchronize());

        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            printf("Error on GPU %d: %s\n", i, hipGetErrorString(err));
            exit(1);
        }
    }
}

void rccl_cleanup(RCCLContext* ctx) {
    if (ctx == NULL) return;

    for (int i = 0; i < ctx->num_gpus; i++) {
        ncclCommDestroy(ctx->comms[i]);
        CHECK_HIP(hipStreamDestroy(ctx->streams[i]));
    }

    free(ctx->device_list);
    free(ctx->comms);
    free(ctx->streams);
    free(ctx);
}
