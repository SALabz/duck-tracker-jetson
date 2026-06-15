#pragma once

class PIDController {
public:
    PIDController(float kp, float ki, float kd);
    
    // Compute PID output given error and time delta
    float compute(float error, float dt);
    
    // Reset integral and previous error
    void reset();

private:
    float kp_;
    float ki_;
    float kd_;
    float integral_;
    float prev_error_;
};