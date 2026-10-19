# CUDA-Accelerated-Semiconductor-Defect-Inspection
https://zenodo.org/records/10715190?utm_source=chatgpt.com
<img width="960" height="541" alt="image" src="https://github.com/user-attachments/assets/5248f089-b988-4e98-8da3-7e661780d105" />

**Classical Computer Vision | CUDA/C++ | GPU Performance Optimization | Semiconductor Inspection**

## Overview

This project aims to develop a GPU-accelerated semiconductor defect inspection pipeline using **classical image processing algorithms implemented in C++ and CUDA**.

The project uses the **Carinthia-S semiconductor SEM image dataset** to investigate defect detection accuracy and optimize computational performance.

Rather than relying on deep learning models, the primary focus is on traditional computer vision techniques, GPU kernel optimization, and reproducible performance benchmarking.

The ultimate goal is to demonstrate how CUDA optimization can improve the throughput and latency of industrial image inspection workloads.

## Objectives

- Develop a semiconductor defect inspection pipeline using classical computer vision algorithms.
- Detect and localize defects in SEM images.
- Implement CPU and CUDA versions of computationally intensive operations.
- Optimize GPU memory access, kernel execution, and CPU–GPU data transfers.
- Evaluate defect detection accuracy against ground-truth annotations.
- Benchmark latency, throughput, and CPU-to-GPU speedup.
- Analyze performance bottlenecks using NVIDIA Nsight Systems and Nsight Compute.

## Dataset

**Carinthia-S — Semiconductor SEM Defect Dataset**

Source: https://zenodo.org/records/16895427

The dataset contains semiconductor scanning electron microscopy (SEM) images with annotated defect regions.

The provided annotations will be used to evaluate defect localization and segmentation performance.

The dataset will be downloaded separately and will not be included directly in this repository unless its redistribution terms permit it.

## Planned Inspection Pipeline

```text
SEM Image
    |
    v
Image Preprocessing
    - Grayscale normalization
    - Gaussian / Median filtering
    - Contrast enhancement
    |
    v
Defect Candidate Detection
    - Gradient / Edge detection
    - Thresholding
    - Morphological operations
    - Template / Reference comparison (if applicable)
    |
    v
Defect Region Analysis
    - Connected components
    - Region filtering
    - Geometric feature extraction
    |
    v
Defect Localization
    - Defect masks
    - Bounding boxes
    - Inspection results
    |
    v
Performance Evaluation
    - Detection accuracy
    - CPU / CUDA benchmarking
    - Nsight profiling
```

The final inspection strategy will be selected based on exploratory analysis of the dataset and the characteristics of each defect category.

## CUDA Optimization Strategy

The project will compare multiple implementations of selected image processing operations.

| Implementation | Description |
|---|---|
| CPU Baseline | Reference C++ implementation |
| CUDA Naive | Basic parallel GPU implementation |
| CUDA Optimized | Performance-tuned CUDA kernels |
| CUDA Pipeline | Integrated GPU processing pipeline |

Potential optimization techniques include:

- Coalesced global memory access
- Shared memory tiling
- Constant memory for small filter kernels
- Kernel fusion
- Reduction optimization
- Asynchronous memory transfers
- CPU–GPU pipeline overlap
- CUDA stream-based execution

Each technique will be evaluated experimentally rather than assumed to improve performance.

## Evaluation Metrics

### Defect Detection

- Precision, Recall, and F1-score
- Intersection over Union (IoU)
- Defect localization quality
- False positive and false negative rates

### GPU Performance

- Kernel execution latency
- End-to-end processing latency
- Images processed per second
- CPU-to-GPU speedup
- GPU memory usage
- Host-to-device and device-to-host transfer overhead

Benchmarks will use identical inputs and equivalent processing logic where possible. Output correctness will be checked before comparing execution performance.

## Development Roadmap

- [ ] Explore Carinthia-S images and annotations
- [ ] Analyze defect types and image characteristics
- [ ] Implement a C++ CPU inspection baseline
- [ ] Implement initial CUDA kernels
- [ ] Validate CUDA results against CPU outputs
- [ ] Optimize memory access and kernel execution
- [ ] Integrate the complete defect inspection pipeline
- [ ] Evaluate detection accuracy
- [ ] Benchmark CPU and CUDA performance
- [ ] Profile bottlenecks with NVIDIA Nsight
- [ ] Document results and optimization findings

## Planned Development Environment

- **Language:** C++ / CUDA
- **GPU:** NVIDIA GeForce RTX 4070 SUPER
- **Platform:** Windows 11 / WSL2 Ubuntu 24.04
- **Build System:** CMake
- **Profiling:** NVIDIA Nsight Systems / Nsight Compute
- **Dataset:** Carinthia-S

The exact CUDA toolkit, compiler, and library versions will be documented after the environment is configured.

## Project Status

**Planning / Initial Development**

This repository currently documents the proposed implementation and evaluation plan. CUDA kernels, defect detection results, and performance benchmarks will be added as development progresses.

No performance improvements or detection accuracy results are claimed at this stage.

## References

- [Carinthia-S Dataset — Zenodo](https://zenodo.org/records/16895427)
- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
