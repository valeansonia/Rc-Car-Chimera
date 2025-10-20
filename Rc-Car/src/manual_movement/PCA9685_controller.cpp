#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>
//#include <Adafruit_PCA9685.h>

int main() {
    const char* i2c_device = "/dev/i2c-1";  // Jetson default I2C bus, confirm with ls /dev/i2c-*
    const int pca9685_addr = 0x50;          // Default PCA9685 address

    // Open I2C device
    int file = open(i2c_device, O_RDWR);
    if (file < 0) {
        std::cerr << "Failed to open I2C bus " << i2c_device << "\n";
        return 1;
    }

    // Specify the address of the I2C Slave to communicate with
    if (ioctl(file, I2C_SLAVE, pca9685_addr) < 0) {
        std::cerr << "Failed to set I2C address 0x50\n";
        close(file);
        return 1;
    }

    // Register to read: MODE1 register (0x00)
    unsigned char reg = 0x00;
    if (write(file, &reg, 1) != 1) {
        std::cerr << "Failed to write register address to PCA9685\n";
        close(file);
        return 1;
    }

    // Read 1 byte from the register
    unsigned char data;
    if (read(file, &data, 1) != 1) {
        std::cerr << "Failed to read data from PCA9685\n";
        close(file);
        return 1;
    }

    std::cout << "PCA9685 MODE1 register value: 0x" << std::hex << (int)data << std::dec << "\n";

    close(file);
    return 0;
}
