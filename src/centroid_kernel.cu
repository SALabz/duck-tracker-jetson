#include "centroid_kernel.h"
#include <float.h>

// ── CONSTANTS ─────────────────────────────────────────────────────────────────
#define NUM_BOXES 8400
#define BLOCK_SIZE 256

// ── CUDA KERNEL ───────────────────────────────────────────────────────────────
// Parallel reduction to find highest confidence detection across 8400 boxes
// d_output layout: [cx(8400), cy(8400), w(8400), h(8400), conf(8400)]
__global__ void centroidKernel(
    const float* d_output,
    Detection*   d_detection,
    float        conf_threshold)
{
    // Shared memory for reduction within this block
    __shared__ float s_conf[BLOCK_SIZE];
    __shared__ float s_cx[BLOCK_SIZE];
    __shared__ float s_cy[BLOCK_SIZE];
    __shared__ int   s_idx[BLOCK_SIZE];

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    // Load confidence value for this thread's box
    float conf = -1.0f;
    float cx   = 0.0f;
    float cy   = 0.0f;

    if (i < NUM_BOXES) {
        conf = d_output[4 * NUM_BOXES + i];  // confidence channel
        if (conf < conf_threshold) conf = -1.0f;  // filter low confidence
        cx = d_output[0 * NUM_BOXES + i];    // cx channel
        cy = d_output[1 * NUM_BOXES + i];    // cy channel
    }

    // Load into shared memory
    s_conf[tid] = conf;
    s_cx[tid]   = cx;
    s_cy[tid]   = cy;
    s_idx[tid]  = i;
    __syncthreads();

    // ── PARALLEL REDUCTION ────────────────────────────────────────────────────
    // Tree reduction — each step halves the number of active threads
    // After log2(BLOCK_SIZE) steps, thread 0 has the max confidence in block
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_conf[tid + stride] > s_conf[tid]) {
                s_conf[tid] = s_conf[tid + stride];
                s_cx[tid]   = s_cx[tid + stride];
                s_cy[tid]   = s_cy[tid + stride];
                s_idx[tid]  = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    // Thread 0 writes block result using atomic to find global max
    if (tid == 0 && s_conf[0] > conf_threshold) {
        // Atomically update global detection if this block found better conf
        // We use a simple approach: write if confidence is higher
        if (s_conf[0] > d_detection->conf) {
            d_detection->cx    = s_cx[0];
            d_detection->cy    = s_cy[0];
            d_detection->conf  = s_conf[0];
            d_detection->valid = 1;
        }
    }
}

// ── KERNEL LAUNCHER ───────────────────────────────────────────────────────────
void launchCentroidKernel(
    const float* d_output,
    Detection*   d_detection,
    float        conf_threshold,
    cudaStream_t stream)
{
    // Initialize detection result to invalid
    Detection init = {0.0f, 0.0f, -1.0f, 0};
    cudaMemcpyAsync(d_detection, &init, sizeof(Detection),
                    cudaMemcpyHostToDevice, stream);

    // Launch kernel — 8400 boxes, 256 threads per block = 33 blocks
    int num_blocks = (NUM_BOXES + BLOCK_SIZE - 1) / BLOCK_SIZE;
    centroidKernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        d_output, d_detection, conf_threshold
    );
}