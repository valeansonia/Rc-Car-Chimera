#include "I2C_addresses.h"
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>

#define I2C_DEV_PATH "/dev/i2c-1"
#define PCA9685_ADDR 0x40

int i2c_fd = -1;

bool initPCA9685() {
    i2c_fd = open(I2C_DEV_PATH, O_RDWR);
    if (i2c_fd < 0) {
        std::cerr << "Failed to open I2C device.\n";
        return false;
    }

    if (ioctl(i2c_fd, I2C_SLAVE, PCA9685_ADDR) < 0) {
        std::cerr << "Failed to connect to PCA9685.\n";
        return false;
    }

    // Wake PCA9685 and set PWM frequency to 50Hz
    uint8_t mode1[2] = {0x00, 0x00}; // MODE1 register
    write(i2c_fd, mode1, 2);
    usleep(500);

    // Set PRE_SCALE for 50Hz (approx 0x79)
    uint8_t prescale[2] = {0xFE, 0x79}; // PRE_SCALE register
    write(i2c_fd, prescale, 2);
    return true;
}

void setPWM(uint8_t channel, uint16_t on, uint16_t off) {
    uint8_t data[5];
    data[0] = 0x06 + 4 * channel; // Base address + offset per channel
    data[1] = on & 0xFF;
    data[2] = on >> 8;
    data[3] = off & 0xFF;
    data[4] = off >> 8;
    write(i2c_fd, data, 5);
}