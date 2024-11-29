Scaling Matrix Multiplication Across Multiple AMD GPUs with RCCL and rocBLAS
============================================================================

.. admonition:: Highlights 

 - **Enhanced Scale**: We extend our previous single-GPU matrix multiplication to handle 32,768 x 32,768 matrices (~12.8 GB of memory) by distributing the computation across 8 GPUs.
  
 - **Performance Improvement**: While a single GPU achieved ~37.5 TFLOPS, our multi-GPU implementation achieves a near-linear speedup, reaching ~280 TFLOPS across 8 GPUs.
  
 - **RCCL Integration**: We leverage AMD's RCCL (ROCm Communication Collectives Library) for efficient GPU-to-GPU communication, specifically using broadcast and allgather operations.
  
 - **Memory Distribution**: The implementation divides matrix A into chunks across GPUs while broadcasting matrix B, demonstrating effective memory management for large-scale computations.

 - **Accuracy Verification**: The implementation maintains the same level of numerical accuracy as our single-GPU version, verified through spot-checking against CPU results.

 This blog post demonstrates how to scale matrix multiplication beyond a single GPU using AMD's RCCL library, showing how to efficiently coordinate multiple GPUs within a single machine for improved performance on larger matrices.

Introduction
------------

In our `previous blog post <https://blog.pebblesandweeds.com/gpu_matmul_blog.html>`_, we implemented matrix multiplication in C using AMD's `rocBLAS <https://rocm.docs.amd.com/projects/rocBLAS/en/latest/>`_ library, specifically utilizing the `rocblas_sgemm <https://rocm.docs.amd.com/projects/rocBLAS/en/latest/reference/level-3.html#rocblas-xgemm-batched-strided-batched>`_ API to leverage AMD's fast GPU `matrix cores <https://www.amd.com/en/technologies/cdna.html>`_. The implementation demonstrated that carefully written C code using rocBLAS could match the performance of PyTorch's highly optimized matrix operations. By implementing the Basic Linear Algebra Subprograms (BLAS) for the `ROCm™ platform <https://www.amd.com/en/products/software/rocm.html>`_, rocBLAS provides the foundational GPU-accelerated matrix multiplication operations that make this performance possible across scientific computing and deep learning applications.

Matrix multiplication is inherently parallelizable - the computation can be efficiently distributed across multiple processing units with minimal dependencies between parallel tasks. Modern servers and supercomputers systems leverage this parallelism by providing multiple GPUs per node, enabling significant computational speedups through parallel execution. While our `single-GPU implementation <https://github.com/pebblesandweeds/gpu_matmul>`_ demonstrated basic rocBLAS capabilities, the parallel nature of matrix multiplication makes it an ideal candidate for multi-GPU execution.

This post extends our previous work by distributing matrix multiplication across multiple GPUs within a single host using `RCCL <https://github.com/ROCmSoftwarePlatform/rccl>`_ (ROCm Communication Collectives Library). RCCL, `documented here <https://rocm.docs.amd.com/projects/rccl/en/latest/>`_, provides efficient communication primitives between GPUs, similar to NVIDIA's NCCL, enabling us to coordinate computation across all available devices to maximize hardware utilization and computational throughput. Our goal is to show how to extend our single-GPU rocBLAS implementation in C to utilize RCCL for coordinating matrix multiplication across multiple GPUs in a single host system.

Multi-GPU Matrix Multiplication: Architecture and Implementation
----------------------------------------------------------------

Single-GPU Matrix Multiplication
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Matrix multiplication on a single GPU leverages highly optimized BLAS libraries like rocBLAS to compute the product of two matrices. For matrices A (M×K) and B (K×N), rocBLAS implements the general matrix multiplication formula:

C = α · op(A) · op(B) + β · C

where op(X) represents optional transpose operations on the input matrices. While efficient for moderate sizes, this approach becomes memory-bound as matrices grow larger, ultimately hitting the memory capacity limits of a single GPU.

.. figure:: _static/single-gpu-flow.png
   :alt: Single GPU Matrix Multiplication Workflow
   :align: center
   
   Workflow of simple matrix multiplication on single GPU

Distributed Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^

To overcome these limitations, we can distribute the computation across multiple GPUs within a single host machine. As shown in the diagram below, our approach splits matrix A horizontally across available GPUs while broadcasting matrix B to each device. This strategy allows each GPU to compute a portion of the final result independently, after which RCCL (ROCm Communication Collectives Library) consolidates these partial results into the complete output matrix using its allGather operation.

.. figure:: _static/matmul_rccl_workflow.png
   :alt: Distributed Matrix Multiplication Workflow
   :align: center
   
   Workflow of distributed matrix multiplication across multiple GPUs

The workflow begins by dividing matrix A into equal horizontal partitions, with each GPU receiving M/n rows for n GPUs. Matrix B is broadcast in its entirety to all GPUs using RCCL's broadcast operation, enabling each device to compute its assigned portion of the output matrix. Once local computation is complete, RCCL's allGather operation efficiently combines these partial results into the final M×N output matrix.

Distribution Strategy
^^^^^^^^^^^^^^^^^^^^^
The distribution phase involves two key operations:

1. **Horizontal Partitioning**: For a system with n GPUs, matrix A is divided into n horizontal sections, each containing M/n rows. This partitioning ensures balanced computation across devices.
2. **Matrix B Broadcasting**: The entire matrix B is replicated across all GPUs. While this requires additional memory, it eliminates the need for inter-GPU communication during the computation phase.

RCCL Communication
^^^^^^^^^^^^^^^^^^
The ROCm Communication Collectives Library (RCCL) provides efficient multi-GPU communication primitives. In our implementation, we utilize:

1. **Broadcast**: For distributing matrix B across all devices, ensuring each GPU has a complete copy for computation
2. **AllGather**: For collecting and combining partial results into the final matrix, efficiently consolidating the distributed computations

Memory Requirements
^^^^^^^^^^^^^^^^^^^

With N = 32,768, each matrix has 1,073,741,824 elements. Using 32-bit floating-point precision:

.. math::

    \text{Per matrix size} = 32,768 \times 32,768 \times 4 \text{ bytes} \approx 4.29 \text{ GB}
    \text{Total memory (3 matrices)} \approx 12.87 \text{ GB}

By distributing across 8 GPUs, each GPU handles:

- 1/8th of Matrix A: ~536 MB
- Full copy of Matrix B: ~4.29 GB
- 1/8th of result Matrix C: ~536 MB

RCCL Integration
^^^^^^^^^^^^^^^^

RCCL provides several collective operations for multi-GPU communication. Our implementation primarily uses two:

1. **Broadcast**: Distributes Matrix B to all GPUs

.. code-block:: c

    // Broadcasting matrix B to all GPUs
    rccl_broadcast_matrix(rccl_ctx, d_B, N * N);

2. **AllGather**: Combines partial results into the final matrix

.. code-block:: c

    // Gathering results from all GPUs
    rccl_gather_matrix_chunks(rccl_ctx, d_C_chunks, d_C_final, chunk_size * N);

Key Implementation Components
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **RCCL Context Setup**

.. code-block:: c

    // Initialize RCCL context
    RCCLContext* rccl_ctx = rccl_init(num_gpus);

2. **Memory Allocation and Data Distribution**

.. code-block:: c

    size_t chunk_size = N / num_gpus;
    size_t chunk_bytes = chunk_size * N * sizeof(float);

    for (int i = 0; i < num_gpus; i++) {
        CHECK_HIP(hipSetDevice(i));
        CHECK_HIP(hipMalloc(&d_A_chunks[i], chunk_bytes));
        CHECK_HIP(hipMalloc(&d_B[i], full_size));
        CHECK_HIP(hipMalloc(&d_C_chunks[i], chunk_bytes));
    }

3. **Parallel Matrix Multiplication**

.. code-block:: c

    CHECK_ROCBLAS(rocblas_sgemm(handles[i],
                           rocblas_operation_none,
                           rocblas_operation_none,
                           N, chunk_size, N,
                           &alpha,
                           d_B[i], N,
                           d_A_chunks[i], N,
                           &beta,
                           d_C_chunks[i], N));

Performance Analysis
--------------------

Benchmark Results
^^^^^^^^^^^^^^^^^

Running on 8 AMD MI250X GPUs, we achieved:
- First run: ~35 TFLOPS per GPU (initialization overhead)
- Subsequent runs: ~35-36 TFLOPS per GPU
- Total system performance: ~280 TFLOPS

Example output:

.. code-block:: text

    GPU 0, Run 1: Time: 234.42 ms, Performance: 35.52 TFLOPS
    GPU 1, Run 1: Time: 234.38 ms, Performance: 35.53 TFLOPS
    ...
    GPU 7, Run 1: Time: 234.45 ms, Performance: 35.51 TFLOPS

Scaling Efficiency
^^^^^^^^^^^^^^^^^^

The implementation shows near-linear scaling across GPUs:
- Single GPU: ~37.5 TFLOPS
- 8 GPUs: ~280 TFLOPS (93.75% scaling efficiency)

Communication Overhead
^^^^^^^^^^^^^^^^^^^^^^

RCCL operations add minimal overhead:
- Broadcast of Matrix B: ~10ms
- AllGather of results: ~15ms

These overheads are negligible compared to the computation time (~234ms per multiplication).

Conclusion
----------

Our multi-GPU implementation successfully scales matrix multiplication across 8 GPUs, enabling processing of larger matrices while maintaining high performance. The near-linear speedup demonstrates the effectiveness of RCCL for GPU communication and our chunk-based distribution strategy.

Key takeaways:
1. RCCL enables efficient multi-GPU coordination with minimal overhead
2. Proper data distribution is crucial for balanced GPU utilization
3. rocBLAS performance scales well across multiple GPUs

This implementation provides a foundation for handling even larger matrices and could be extended to multi-node configurations using technologies like ROCm-aware MPI.

For the complete implementation, check out our `GitHub repository <link>`_.
