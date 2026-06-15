#include "serial_control.h"
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <iostream>
#include <cstring>

SerialControl::SerialControl(const std::string& port, int baud_rate)
    : port_(port), baud_rate_(baud_rate), fd_(-1), current_angle_(90) {
    if (!openPort()) {
        std::cerr << "Failed to open serial port: " << port << std::endl;
    } else {
        configurePort();
        sleep(2);  // wait for Arduino to initialize
        std::cout << "Serial port opened: " << port << std::endl;
        sendAngle(90);  // center servo on startup
    }
}

SerialControl::~SerialControl() {
    close();
}

bool SerialControl::openPort() {
    fd_ = open(port_.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
    return fd_ >= 0;
}

void SerialControl::configurePort() {
    struct termios tty;
    memset(&tty, 0, sizeof tty);
    tcgetattr(fd_, &tty);

    cfsetospeed(&tty, B9600);
    cfsetispeed(&tty, B9600);

    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_iflag &= ~IGNBRK;
    tty.c_lflag = 0;
    tty.c_oflag = 0;
    tty.c_cc[VMIN]  = 0;
    tty.c_cc[VTIME] = 5;
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~(PARENB | PARODD);
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;

    tcsetattr(fd_, TCSANOW, &tty);
}

void SerialControl::sendAngle(int angle) {
    if (fd_ < 0) return;
    angle = std::max(0, std::min(180, angle));
    current_angle_ = angle;
    std::string cmd = std::to_string(angle) + "\n";
    write(fd_, cmd.c_str(), cmd.length());
}

void SerialControl::sendCommand(const std::string& command) {
    if (fd_ < 0) return;
    std::string cmd = command + "\n";
    write(fd_, cmd.c_str(), cmd.length());
}

bool SerialControl::isOpen() const {
    return fd_ >= 0;
}

void SerialControl::close() {
    if (fd_ >= 0) {
        sendAngle(90);  // center before closing
        ::close(fd_);
        fd_ = -1;
        std::cout << "Serial port closed" << std::endl;
    }
}