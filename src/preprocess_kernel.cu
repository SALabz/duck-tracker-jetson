#include "preprocess_kernel.h"

// ── CUDA KERNEL ───────────────────────────────────────────────────────────────
// Each thread handles one output pixel
// Performs: bilinear resize + BGR->RGB + normalize + HWC->CHW
__global__ void preprocessKernel(
    const unsigned char* src,  // input BGR frame (HWC)
    float* dst,                // output float32 (CHW)
    int srcW, int srcH,
    int dstW, int dstH)
{
    // Compute output pixel coordinates from thread/block index
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Bounds check
    if (dst_x >= dstW || dst_y >= dstH) return;

    // Map output pixel back to input pixel (scale factors)
    float scale_x = (float)srcW / (float)dstW;
    float scale_y = (float)srcH / (float)dstH;

    float src_x = dst_x * scale_x;
    float src_y = dst_y * scale_y;

    // Bilinear interpolation
    int x0 = (int)src_x;
    int y0 = (int)src_y;
    int x1 = min(x0 + 1, srcW - 1);
    int y1 = min(y0 + 1, srcH - 1);

    float wx = src_x - x0;  // fractional part
    float wy = src_y - y0;

    // Sample 4 neighboring pixels (BGR format, 3 channels)
    const unsigned char* p00 = src + (y0 * srcW + x0) * 3;
    const unsigned char* p01 = src + (y0 * srcW + x1) * 3;
    const unsigned char* p10 = src + (y1 * srcW + x0) * 3;
    const unsigned char* p11 = src + (y1 * srcW + x1) * 3;

    // Bilinear interpolate each channel
    // Note: BGR input → RGB output (swap channels 0 and 2)
    float pixels[3];
    for (int c = 0; c < 3; c++) {
        int src_c = (c == 0) ? 2 : (c == 2) ? 0 : 1;  // BGR->RGB swap

        float top    = p00[src_c] * (1 - wx) + p01[src_c] * wx;
        float bottom = p10[src_c] * (1 - wx) + p11[src_c] * wx;
        pixels[c] = top * (1 - wy) + bottom * wy;
    }

    // Normalize to [0, 1] and write in CHW layout
    int channel_size = dstW * dstH;
    int pixel_idx    = dst_y * dstW + dst_x;

    dst[0 * channel_size + pixel_idx] = pixels[0] / 255.0f;  // R
    dst[1 * channel_size + pixel_idx] = pixels[1] / 255.0f;  // G
    dst[2 * channel_size + pixel_idx] = pixels[2] / 255.0f;  // B
}

// ── KERNEL LAUNCHER ───────────────────────────────────────────────────────────
void launchPreprocessKernel(
    const unsigned char* src,
    float* dst,
    int srcW, int srcH,
    int dstW, int dstH,
    cudaStream_t stream)
{
    // 16x16 thread block — each thread handles one output pixel
    dim3 block(16, 16);
    dim3 grid(
        (dstW + block.x - 1) / block.x,
        (dstH + block.y - 1) / block.y
    );

    preprocessKernel<<<grid, block, 0, stream>>>(
        src, dst, srcW, srcH, dstW, dstH
    );
}