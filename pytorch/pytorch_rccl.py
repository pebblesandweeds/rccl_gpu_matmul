# Run with `torchrun --nproc_per_node=8 pytorch_rccl.py` in a Python virtual env installed with ROCm torch

import torch
import torch.distributed as dist
import time

def init_process():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size

def main():
    rank, world_size = init_process()
    N = 32768
    NUM_RUNS = 25
    chunk_size = N // world_size
    device = torch.device(f'cuda:{rank}')

    if rank == 0:
        print(f"Matrix size: {N}x{N}")
        print(f"Running on {world_size} GPUs\n")
        for i in range(world_size):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Create my chunk of matrix A
    my_chunk = torch.empty(chunk_size, N, dtype=torch.float32, device=device).uniform_(-1, 1)

    # Only rank 0 creates matrix B initially
    if rank == 0:
        B = torch.empty(N, N, dtype=torch.float32, device=device).uniform_(-1, 1)
    else:
        B = torch.empty(N, N, dtype=torch.float32, device=device)

    if rank == 0:
        print("\nBroadcasting matrix B to all GPUs")

    # Broadcast B to all GPUs
    dist.broadcast(B, src=0)

    # Calculate FLOPS for one GPU's portion
    flops = (2 * N * N * N) / world_size

    if rank == 0:
        print("Starting matrix multiplication runs...")

    for run in range(NUM_RUNS):
        torch.cuda.synchronize()
        dist.barrier()

        start = time.perf_counter()
        result = torch.matmul(my_chunk, B)
        torch.cuda.synchronize()
        end = time.perf_counter()

        run_time = (end - start) * 1000  # Convert to ms
        tflops = (flops / (run_time / 1000)) / 1e12

        print(f"GPU {rank}, Run {run + 1}: Time: {run_time:.2f} ms, Performance: {tflops:.2f} TFLOPS")
        dist.barrier()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
