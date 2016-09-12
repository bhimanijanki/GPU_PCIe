# GPU_PCIe
Design Space Exploration of GPU Accelerated Cluster Systems for Optimal Data Transfer Using PCIe Bus

27 version of micro benchmark [1], with each making different design choices.

- micro_bench_pin_calB.cu
  - micro benchmark using pinned memorory allocation
  - calculation bound
  - contains 3 different kernels
      - no cuda stream
      - cuda stream with type I scheduling scheme
      - cuda stream with type II scheduling scheme
- micro_bench_pin_comB.cu
  - micro benchmark using pinned memorory allocation
  - communication bound
  - contains 3 different kernels
      - no cuda stream
      - cuda stream with type I scheduling scheme
      - cuda stream with type II scheduling scheme
- micro_bench_pin_balB.cu
  - micro benchmark using pinned memorory allocation
  - balanced bound
  - contains 3 different kernels
      - no cuda stream
      - cuda stream with type I scheduling scheme
      - cuda stream with type II scheduling scheme

- micro_bench_pageable_calB.cu
  - micro benchmark using pageable memorory allocation
  - calculation bound
  - contains 3 different kernels
      - no cuda stream
      - cuda stream with type I scheduling scheme
      - cuda stream with type II scheduling scheme
- micro_bench_pageable_comB.cu
  - micro benchmark using pageable memorory allocation
  - communication bound
  - contains 3 different kernels
      - no cuda stream
      - cuda stream with type I scheduling scheme
      - cuda stream with type II scheduling scheme
- micro_bench_pin_balB.cu
  - micro benchmark using pageable memorory allocation
  - balanced bound
  - contains 3 different kernels
      - no cuda stream
      - cuda stream with type I scheduling scheme
      - cuda stream with type II scheduling scheme

- micro_bench_UM_calB.cu
  - micro benchmark using unified memorory allocation
  - calculation bound
  - contains 3 different kernels
      - no cuda stream
      - cuda stream with type I scheduling scheme
      - cuda stream with type II scheduling scheme
- micro_bench_UM_comB.cu
  - micro benchmark using unified memorory allocation
  - communication bound
  - contains 3 different kernels
      - no cuda stream
      - cuda stream with type I scheduling scheme
      - cuda stream with type II scheduling scheme
- micro_bench_UM_balB.cu
  - micro benchmark using unified memorory allocation
  - balanced bound
  - contains 3 different kernels
      - no cuda stream
      - cuda stream with type I scheduling scheme
      - cuda stream with type II scheduling scheme

[1] M. Harris, “How to overlap data transfers in CUDA C/C++,” Dec. 2012. [Online]. 
Available: https://devblogs.nvidia.com/parallelforall/how-overlap-data-transfers-cuda-cc/
