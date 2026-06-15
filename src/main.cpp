#include <iostream>
#include <chrono>
#include <thread>
#include <signal.h>

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include "pid_controller.h"
#include "serial_control.h"
#include "preprocess_kernel.h"
#include "centroid_kernel.h"

// ── CONFIG ────────────────────────────────────────────────────────────────────
#define ENGINE_PATH     "/home/seraj/runs/detect/duck_detector/weights/best_cpp.engine"
#define SERIAL_PORT     "/dev/ttyACM0"
#define BAUD_RATE       9600
#define FRAME_WIDTH     640
#define FRAME_HEIGHT    480
#define CAMERA_INDEX    0
#define SERVO_MIN       0
#define SERVO_MAX       180
#define SERVO_CENTER    90
#define CONF_THRESHOLD  0.5f

// PID gains — tune these
#define KP  0.03f
#define KI  0.0f
#define KD  0.0f

// ── GLOBALS ───────────────────────────────────────────────────────────────────
volatile bool running = true;

void signalHandler(int sig) {
    std::cout << "\nShutting down..." << std::endl;
    running = false;
}

// ── TENSORRT LOGGER ───────────────────────────────────────────────────────────
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

// ── MAIN ──────────────────────────────────────────────────────────────────────
int main() {
    signal(SIGINT, signalHandler);

    // ── SERIAL ────────────────────────────────────────────────────────────────
    SerialControl servo(SERIAL_PORT, BAUD_RATE);
    if (!servo.isOpen()) {
        std::cerr << "Failed to open serial port" << std::endl;
        return -1;
    }

    // ── CAMERA ────────────────────────────────────────────────────────────────
    cv::VideoCapture cap(CAMERA_INDEX);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open camera" << std::endl;
        return -1;
    }
    std::cout << "Camera opened at " << FRAME_WIDTH << "x" << FRAME_HEIGHT << std::endl;

    // ── PID ───────────────────────────────────────────────────────────────────
    PIDController pid(KP, KI, KD);
    int current_angle = SERVO_CENTER;
    float frame_center_x = FRAME_WIDTH / 2.0f;

    // ── TENSORRT ENGINE ───────────────────────────────────────────────────────
    std::cout << "Loading TensorRT engine..." << std::endl;

    // Read engine file
    FILE* f = fopen(ENGINE_PATH, "rb");
    if (!f) {
        std::cerr << "Failed to open engine file" << std::endl;
        return -1;
    }
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> engineData(size);
    fread(engineData.data(), 1, size, f);
    fclose(f);

    // Deserialize engine
    auto runtime = nvinfer1::createInferRuntime(gLogger);
    auto engine  = runtime->deserializeCudaEngine(engineData.data(), size);
    auto context = engine->createExecutionContext();

    std::cout << "TensorRT engine loaded" << std::endl;

    // ── CUDA BUFFERS ──────────────────────────────────────────────────────────
    // Input: [1, 3, 640, 640] float32
    // Output: [1, 5, 8400] float32
    int input_size  = 1 * 3 * 640 * 640 * sizeof(float);
    int output_size = 1 * 5 * 8400 * sizeof(float);

    void* d_input;
    void* d_output;
    cudaMalloc(&d_input,  input_size);
    cudaMalloc(&d_output, output_size);

    // Raw frame buffer on GPU (uint8 BGR)
    unsigned char* d_raw_frame;
    int raw_frame_size = FRAME_WIDTH * FRAME_HEIGHT * 3 * sizeof(unsigned char);
    cudaMalloc(&d_raw_frame, raw_frame_size);
    // Detection result buffer on GPU
    Detection* d_detection;
    cudaMalloc(&d_detection, sizeof(Detection));

    std::vector<float> h_input(1 * 3 * 640 * 640);
    std::vector<float> h_output(1 * 5 * 8400);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // ── BENCHMARK SETUP ───────────────────────────────────────────────────────
    int frame_count = 0;
    double total_preprocess_ms = 0;
    double total_inference_ms  = 0;
    double total_postprocess_ms = 0;

    std::cout << "Duck tracker running. Press Ctrl+C to stop." << std::endl;

    // ── MAIN LOOP ─────────────────────────────────────────────────────────────
    auto prev_time = std::chrono::steady_clock::now();

    while (running) {
        cv::Mat frame;
        if (!cap.read(frame)) {
            std::cerr << "Camera read failed" << std::endl;
            break;
        }

        auto loop_start = std::chrono::steady_clock::now();

        // ── PREPROCESS (CPU baseline — will be replaced by CUDA plugin) ───────
        auto t0 = std::chrono::steady_clock::now();

        // Upload raw BGR frame to GPU
        cudaMemcpyAsync(d_raw_frame, frame.data, raw_frame_size,
                        cudaMemcpyHostToDevice, stream);

        // Launch CUDA preprocessing kernel
        launchPreprocessKernel(
            d_raw_frame,
            (float*)d_input,
            FRAME_WIDTH, FRAME_HEIGHT,
            640, 640,
            stream
        );

        cudaStreamSynchronize(stream);

        auto t1 = std::chrono::steady_clock::now();
        total_preprocess_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

        // ── INFERENCE ─────────────────────────────────────────────────────────
        void* bindings[] = {d_input, d_output};
        context->executeV2(bindings);
        cudaStreamSynchronize(stream);

        auto t2 = std::chrono::steady_clock::now();
        total_inference_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();

        // ── POSTPROCESS ───────────────────────────────────────────────────────
        // Launch GPU centroid extraction kernel
        launchCentroidKernel(
            (const float*)d_output,
            d_detection,
            CONF_THRESHOLD,
            stream
        );
        cudaStreamSynchronize(stream);

        // Copy result back to CPU (just 4 floats — minimal transfer)
        Detection h_detection;
        cudaMemcpy(&h_detection, d_detection, sizeof(Detection),
                cudaMemcpyDeviceToHost);

        auto t3 = std::chrono::steady_clock::now();
        total_postprocess_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();

        // ── PID + SERVO ───────────────────────────────────────────────────────
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - prev_time).count();
        prev_time = now;

        if (h_detection.valid) {
            float error = h_detection.cx - frame_center_x;
            float correction = pid.compute(error, dt);
            current_angle = std::max(SERVO_MIN,
                            std::min(SERVO_MAX,
                            (int)(current_angle - correction)));
            servo.sendAngle(current_angle);

            std::cout << "Duck | cx=" << h_detection.cx
                    << " error=" << error
                    << " angle=" << current_angle
                    << " conf=" << h_detection.conf << std::endl;
        } else {
            std::cout << "No duck detected" << std::endl;
        }

        frame_count++;

        // Print benchmark every 100 frames
        if (frame_count % 100 == 0) {
            std::cout << "\n── BENCHMARK (avg over " << frame_count << " frames) ──" << std::endl;
            std::cout << "Preprocess : " << total_preprocess_ms  / frame_count << " ms" << std::endl;
            std::cout << "Inference  : " << total_inference_ms   / frame_count << " ms" << std::endl;
            std::cout << "Postprocess: " << total_postprocess_ms / frame_count << " ms" << std::endl;
            std::cout << "────────────────────────────────────────\n" << std::endl;
        }
    }

    // ── CLEANUP ───────────────────────────────────────────────────────────────
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_raw_frame);
    cudaFree(d_detection);
    cudaStreamDestroy(stream);
    delete context;
    delete engine;
    delete runtime;

    std::cout << "Done." << std::endl;
    return 0;
}