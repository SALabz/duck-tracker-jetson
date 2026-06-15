#pragma once
#include <cuda_runtime.h>

// Launch the preprocessing kernel
// src: raw BGR frame from camera (on GPU)
// dst: normalized CHW float32 output for TensorRT (on GPU)
// srcW, srcH: input frame dimensions
// dstW, dstH: output dimensions (640x640)
void launchPreprocessKernel(
    const unsigned char* src,
    float* dst,
    int srcW, int srcH,
    int dstW, int dstH,
    cudaStream_t stream
);