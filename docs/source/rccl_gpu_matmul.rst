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

In our `previous blog post <https://blog.pebblesandweeds.com/gpu_matmul_blog.html>`_, we implemented matrix multiplication in C using AMD's `rocBLAS <https://rocm.docs.amd.com/projects/rocBLAS/en/latest/>`_ library, specifically utilizing the `rocblas_sgemm <https://rocm.docs.amd.com/projects/rocBLAS/en/latest/reference/level-3.html#rocblas-xgemm-batched-strided-batched>`_ API to leverage AMD's fast GPU `matrix cores <https://www.amd.com/en/technologies/cdna.html>`_. The implementation demonstrated that carefully written C code using rocBLAS could match the performance of PyTorch's highly optimized matrix operations. By leveraging rocBLAS's BLAS (Basic Linear Algebra Subprograms) implementation for the ROCm platform, we achieved GPU-accelerated matrix multiplication performance that matches PyTorch for our deep learning applications. 

While our previous work focused on single-GPU matrix multiplication, this operation is inherently parallelizable - computations can be efficiently distributed across multiple processing units with minimal dependencies between parallel tasks. Modern servers and supercomputers systems support this parallelism by providing multiple GPUs per node, enabling significant computational speedups through parallel execution. While our `single-GPU implementation <https://github.com/pebblesandweeds/gpu_matmul>`_ demonstrated basic rocBLAS capabilities, the parallel nature of matrix multiplication makes it an ideal candidate for multi-GPU execution.

This post extends our previous work by distributing matrix multiplication across multiple GPUs within a single host using `RCCL <https://github.com/ROCmSoftwarePlatform/rccl>`_ (ROCm Communication Collectives Library). `RCCL provides <https://rocm.docs.amd.com/projects/rccl/en/latest/>`_ efficient communication primitives between GPUs, similar to NVIDIA's NCCL, enabling us to coordinate computation across all available devices to maximize hardware utilization and computational throughput. Our goal is to show how to extend our single-GPU rocBLAS implementation in C to utilize RCCL for coordinating matrix multiplication across multiple GPUs in a single host system.

Scaling Matrix Multiplication: From Single to Multi-GPU Systems
----------------------------------------------------------------

Single-GPU Matrix Multiplication
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The rocBLAS ``sgemm`` API implements high-performance single precision (fp32) matrix multiplication using AMD's matrix core accelerators (detailed formula and optimizations covered in our `previous post <https://blog.pebblesandweeds.com/gpu_matmul_blog.html#matrix-multiplication-formulas>`_). The core workflow involves transferring input matrices A and B to GPU memory, executing the multiplication, and transferring result matrix C back to host memory.

While this appears straightforward, achieving peak performance requires careful orchestration of memory transfers, matrix layouts, and compute scheduling. Thankfully, rocBLAS abstracts away many of these complexities - it handles matrix padding and alignment to maximize memory throughput, manages optimal blocking strategies for AMD's matrix cores, and provides batching capabilities for efficient execution of multiple multiplications. This allows developers to focus on high-level algorithm design while the library manages the hardware-specific optimizations.

Even though this single-GPU approach delivers good performance for matrices that fit within GPU memory, it is ultimately constrained by both memory capacity and computational throughput of a single device. A modern GPU can deliver impressive TFLOP/s for matrix operations, but most AI workloads demand higher computational capabilities than a single GPU can deliver. These performance demands, combined with memory limitations, motivate exploration of multi-GPU approaches that can harness both the aggregate compute power and memory capacity of multiple devices.

.. figure:: _static/single-gpu-flow.png
  :alt: Single GPU Matrix Multiplication Workflow
  :align: center

  Simple matrix multiplication on single GPU

Distributed Matrix Multiplication 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Extending beyond a single device, we can leverage multiple GPUs within a host system to dramatically increase both computational throughput and available memory. The key lies in efficiently partitioning the workload while minimizing data transfers between devices.

Our distributed implementation employs a horizontal partitioning strategy that balances computational efficiency with communication overhead through several key mechanisms:

* **Matrix Distribution** - Matrix A is split horizontally across GPUs while matrix B is broadcast in its entirety to each device, allowing independent processing of matrix partitions using rocBLAS primitives.

* **Result Consolidation**: The system combines partial results from each device through RCCL's allGather operation, constructing the final output matrix

* **Performance Optimization**: The approach maximizes efficiency through balanced computational load from the horizontal split of A, eliminating the inter-GPU communication by broadcasting B, and minimal overhead during result collection via allGather

Through these design choices, we transform our earlier single-GPU implementation into a scalable distributed system that preserves the computational efficiency of rocBLAS while extending across multiple devices.

.. figure:: _static/matmul_rccl_workflow.png
   :alt: Distributed Matrix Multiplication Workflow
   :align: center

   Distributed matrix multiplication across multiple GPUs

Broadcasting matrix B instead of partitioning it offers advantages for deep learning workloads, despite higher per-GPU memory usage. This approach eliminates inter-GPU communication since matrices A and B serve different purposes: 

* Matrix B represents stable parameters (model weights during inference, parameter gradients during training)
* Matrix A contains the changing data stream (input activations or training examples)
* Computing any element C[i,j] requires both the complete ith row of A and jth column of B

Chunking B would force GPUs to constantly exchange partial results during computation rather than just at initialization and completion. The efficiency comes from keeping a complete copy of B on each device - the initial broadcast cost is offset by being able to reuse B for multiple computations with streaming A matrices. While alternatives like `Cannon's algorithm <https://en.wikipedia.org/wiki/Cannon%27s_algorithm>`_ provide more memory-efficient partitioning, the additional coordination overhead makes broadcasting B preferable given modern GPU memory capacities and deep learning's characteristic reuse of parameter matrices across batches.

Implementing Multi-GPU Matrix Multiplication
--------------------------------------------

Building on our distributed matrix multiplication concepts, this section walks through the practical implementation details. We'll examine how the code coordinates computation across multiple GPUs, diving into the key libraries that enable efficient distribution and the resulting memory patterns across devices.

Implementation Libraries 
^^^^^^^^^^^^^^^^^^^^^^^^
Our implementation leverages two core AMD libraries:

**rocBLAS for Matrix Computation**

The `rocblas_sgemm` function handles the actual matrix multiplication on each GPU. After receiving its chunk of matrix A and complete copy of matrix B, each GPU executes a standard matrix multiplication operation. rocBLAS automatically optimizes this computation for AMD's matrix cores, managing internal memory layouts and compute scheduling.

**RCCL for GPU Communication**

RCCL (ROCm Communication Collectives Library) provides efficient primitives for moving data between GPUs. While this is AMD's library, it maintains API compatibility with NVIDIA's NCCL - hence the `nccl` prefix in function names like `ncclBroadcast`. Our implementation uses two key RCCL operations:

* ``ncclBroadcast`` distributes matrix B to all GPUs during initialization
* ``ncclAllGather`` combines partial results from each GPU's computation into the final output matrix

RCCL handles the complexity of optimal data transfer paths between GPUs, utilizing direct GPU-to-GPU communication when available and automatically selecting the most efficient transfer methods based on system topology.

The interaction between these libraries follows a clear pattern: RCCL first distributes the input data across devices, rocBLAS performs local computations on each GPU, and finally RCCL consolidates the results. This separation of concerns - RCCL for communication and rocBLAS for computation - allows each library to optimize its specific role while working together for efficient distributed processing.

Memory Requirements
^^^^^^^^^^^^^^^^^^^

Let's examine the memory distribution patterns across GPUs in our matrix multiplication implementation. For this discussion, we'll use 32K × 32K matrices with single precision floating point values (fp32, 4 bytes per element). Each complete matrix occupies:

.. math::

   32,768 \times 32,768 \times 4 \text{ bytes} \approx 4.29 \text{ GB}

While these matrices are modest in size for modern enterprise GPUs, they serve as an example for understanding the memory efficiency benefits of distributed computation.

**Single-GPU Memory Footprint**

When running matrix multiplication on a single GPU using rocBLAS (as covered in our previous blog post), we need all three matrices to reside in device memory. With each matrix requiring 4.29 GB, our total VRAM usage is ~12.87 GB for matrices A, B, and C. While this memory footprint is well within the capabilities of modern GPUs, by distributing these matrices across devices we can reduce the per-GPU memory requirements, paving the way for larger computations or processing multiple matrix multiplications in parallel.

**Distributed Memory Layout**

Our 8-GPU implementation reduces per-device memory usage through selective matrix distribution. Each GPU stores:

* 1/8th chunk of matrix A: 4.29 GB ÷ 8 ≈ 536 MB
* Complete copy of matrix B: 4.29 GB
* 1/8th chunk of output matrix C: 536 MB

This distribution strategy requires ~5.36 GB per GPU compared to the 12.87 GB needed for single-GPU execution. The reduction stems from dividing matrices A and C across devices while broadcasting B to each GPU. While in this example our memory savings are modest, this pattern becomes increasingly important when scaling to larger matrices or processing multiple matrix multiplications in parallel.

It's worth noting that in real-world deep learning applications, we typically process batches of matrix multiplications rather than single operations. While batched operations are beyond the scope of this blog post, the memory distribution strategy demonstrated here - chunking A and C while broadcasting B - provides an efficient foundation for handling these larger workloads.

Coordinating GPU Communication with RCCL
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RCCL (ROCm Communication Collectives Library) provides efficient primitives for communication between multiple GPUs in a system. For our matrix multiplication implementation across 8 GPUs in a single host, understanding RCCL's core components and operations is essential.

RCCL operates through communicator objects (ncclComm_t) that represent a collection of GPUs that can communicate with each other. Each GPU is assigned a unique rank (0 to 7 in our case) within the communicator, corresponding to their HIP device IDs. RCCL operations are asynchronous and tied to HIP streams, with each GPU requiring a dedicated stream to ensure proper synchronization.

The library provides several communication primitives, but our implementation focuses on two key operations:

* **Broadcast**: Copies data from one GPU (root) to all other GPUs. We use this to distribute matrix B to all GPUs efficiently, ensuring each device has the complete matrix for computation.

* **AllGather**: Each GPU contributes a chunk of data that is gathered and made available to all GPUs. We use this to combine the partial results of matrix C from each GPU into the complete result matrix.

RCCL automatically leverages dedicated hardware and protocols for GPU-to-GPU communication within our single host system, providing significantly better performance than standard PCIe transfers. The library handles the complexity of selecting optimal data transfer paths between GPUs based on the available hardware.

Since RCCL operations are asynchronous, proper synchronization is necessary. Operations in the same stream execute sequentially, and error checking should be performed after synchronization rather than immediately after RCCL calls. Our implementation includes appropriate error handling and ensures proper cleanup of RCCL resources to prevent memory leaks.

This foundation enables our multi-GPU matrix multiplication to efficiently distribute computation while minimizing the overhead of data movement between devices. The following sections will demonstrate how we implement these concepts in practice.

---------------------------



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
