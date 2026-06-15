#pragma once
#include <string>

class SerialControl {
public:
    SerialControl(const std::string& port, int baud_rate);
    ~SerialControl();

    // Send angle to Arduino (0-180)
    void sendAngle(int angle);

    // Send named command
    void sendCommand(const std::string& command);

    // Check if serial port is open
    bool isOpen() const;

    // Close the port
    void close();

private:
    int fd_;  // file descriptor
    std::string port_;
    int baud_rate_;
    int current_angle_;

    bool openPort();
    void configurePort();
};