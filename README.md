# Distributed Matrix Multiplication using RCCL 

This repository demonstrates distributed matrix multiplication across multiple AMD GPUs using RCCL (ROCm Communication Collectives Library) and rocBLAS. The implementation distributes rocblas_sgemm() matrix multiplication computation across multiple GPUs within a single host system to achieve enhanced performance through parallel processing.

## Overview
The project provides two implementations:

* A C implementation using RCCL and rocBLAS for distributed GPU matrix multiplication on a single host
* A PyTorch implementation using DistributedDataParallel with RCCL backend on a single host

Both implementations demonstrate how to:

* Distribute large matrix operations across multiple GPUs
* Coordinate inter-GPU communication using RCCL
* Demonstrate single host, multi-GPU performance through data distribution and synchronization

## Getting Started

### Prerequisites

* AMD ROCm 6.x
* RCCL (ROCm Communication Collectives Library)
* rocBLAS
* Python 3.x (for PyTorch implementation)
* PyTorch 2.x+rocm6.x with RCCL support

### Installation

Installation of AMD ROCm 6.x (includes RCCL and rocBLAS) and Pytorch with ROCm support are out of scope for this repo, see [AMD documentation](https://github.com/ROCm/ROCm) for instructions.  The only dependency needed for Python is PyTorch with ROCm support, a separate requirements.txt file is unnecessary. 

## Usage

### Running the Pytorch Script

It is recommended to use a Python virtual environment (`venv`, `pyenv`, `conda`) with Pytorch with ROCm support installed.  Run the Pytorch script using `torchrun --nproc_per_node=8 pytorch_rccl.py`.

### Running the C code

1.  Change directories `cd c`
2.  Compile and run the benchmark `make && ./multi_gpu_matmul`

### Performance Output

The PyTorch implementation uses `torch.matmul`, which abstracts away the complexities of GPU interaction, making the code concise and straightforward. When running this version, users can expect it to set up the input tensors and perform matrix multiplication efficiently using the GPU, with minimal manual intervention. The output will provide a summary of the computation time and performance achieved across multiple runs.

The C implementation uses the AMD rocBLAS library directly, requiring more hands-on setup, including GPU context initialization and data transfers. Since we are implementing the rocBLAS framework ourselves, it also includes accuracy checks to ensure the results are correct. This version outputs GPU specifications, memory transfer times, matrix multiplication times, and the performance for each run. Users will also see results of spot checks to confirm the numerical correctness of the GPU computations compared to CPU expectations, which is important to ensure that high performance isn't accompanied by incorrect results.

## Project Structure

```
rccl_gpu_matmul/
├── LICENSE
├── README.md
├── c/
│   ├── Makefile
│   ├── include/
│   │   ├── matrix_operations.h
│   │   ├── rccl_utils.h
│   │   ├── spot_check.h
│   │   ├── timer.h
│   │   └── utils.h
│   └── src/
│       ├── main.c
│       ├── matrix_operations.c
│       ├── rccl_utils.c
│       ├── spot_check.c
│       ├── timer.c
│       └── utils.c
└── pytorch/
    └── pytorch_matmul.py
```
