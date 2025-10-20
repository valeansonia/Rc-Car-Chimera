#include <linux/input.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>

#include "hardcoded_addresses.h"
#include "I2C_addresses.h"

#define DEVICE_PATH "/dev/input/event13"  

// map trigger value (0–255) to PWM (e.g. 0–4095)
uint16_t mapTriggerToPWM(int value) {
    return (uint16_t)((value / 255.0) * 4095);
}


int main() {
    int fd = open(DEVICE_PATH, O_RDONLY);
    if (fd < 0) {
        std::cerr << "Cannot open input device\n";
        return 1;
    }

    if (!initPCA9685()) {
        return 1;
    }

    struct input_event ev;
    while (read(fd, &ev, sizeof(ev)) > 0) {
        if (ev.type == EV_ABS) {
            if (ev.code == XBOX_GAS_TRIGGER) {
                uint16_t pwm = mapTriggerToPWM(ev.value);
                std::cout << "Gas pressed: PWM = " << pwm << std::endl;
                setPWM(0, 0, pwm); // channel 0
            }
            else if (ev.code == XBOX_BRAKE_TRIGGER) {
                int pwm = mapTriggerToPWM(ev.value);
                std::cout << "Brake pressed: PWM = " << pwm << std::endl;
                setPWM(1, 0, pwm); // channel 1
            }
        }
    }

    close(fd);
    return 0;
}