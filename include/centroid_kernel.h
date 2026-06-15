#pragma once
#include <cuda_runtime.h>

struct Detection {
    float cx;
    float cy;
    float conf;
    int   valid;  // 1 if detection found, 0 if not
};

// Launch centroid extraction kernel
// d_output: TensorRT output tensor [1, 5, 8400] on GPU
// d_detection: result struct on GPU
// conf_threshold: minimum confidence to count as detection
void launchCentroidKernel(
    const float* d_output,
    Detection*   d_detection,
    float        conf_threshold,
    cudaStream_t stream
);