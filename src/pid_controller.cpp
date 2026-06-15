#include "pid_controller.h"

PIDController::PIDController(float kp, float ki, float kd)
    : kp_(kp), ki_(ki), kd_(kd), integral_(0.0f), prev_error_(0.0f) {}

float PIDController::compute(float error, float dt) {
    // Proportional
    float p = kp_ * error;

    // Integral
    integral_ += error * dt;
    float i = ki_ * integral_;

    // Derivative
    float derivative = (dt > 0) ? (error - prev_error_) / dt : 0.0f;
    float d = kd_ * derivative;

    prev_error_ = error;

    return p + i + d;
}

void PIDController::reset() {
    integral_ = 0.0f;
    prev_error_ = 0.0f;
}