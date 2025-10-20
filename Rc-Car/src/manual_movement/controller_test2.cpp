#include <iostream>
#include <linux/input.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include "hardcoded_addresses.h"

int main() {
    const char* device = "/dev/input/event13"; 
    int fd = open(device, O_RDONLY);
    if (fd < 0) {
        std::cerr << "Failed to open " << device << "\n";
        return 1;
    }

    std::cout << "Listening for Xbox controller input...\n";

    struct input_event ev;
    while (true) {
        ssize_t bytes = read(fd, &ev, sizeof(ev));
        if (bytes == sizeof(ev)) {
            if ((ev.type == EV_KEY || ev.type == EV_ABS) && ev.value != 0) {
                std::cout << "Code: 0x" << std::hex << ev.code 
                          << " | Value: " << std::dec << ev.value << " → ";

                switch (ev.code) {
                    // triggers
                    case XBOX_GAS_TRIGGER:      std::cout << "Gas Trigger (RT) pressed\n"; break;
                    case XBOX_BRAKE_TRIGGER:    std::cout << "Brake Trigger (LT) pressed\n"; break;

                    // joysticks
                    case XBOX_LEFT_STICK_X:     std::cout << "Left Stick X moved\n"; break;
                    case XBOX_LEFT_STICK_Y:     std::cout << "Left Stick Y moved\n"; break;
                    case XBOX_RIGHT_STICK_X:    std::cout << "Right Stick X moved\n"; break;
                    case XBOX_RIGHT_STICK_Y:    std::cout << "Right Stick Y moved\n"; break;

                    // D-Pad
                    case XBOX_DPAD_X:           std::cout << "D-Pad Left/Right\n"; break;
                    case XBOX_DPAD_Y:           std::cout << "D-Pad Up/Down\n"; break;

                    // face buttons
                    case XBOX_BUTTON_A:         std::cout << "Button A pressed\n"; break;
                    case XBOX_BUTTON_B:         std::cout << "Button B pressed\n"; break;
                    case XBOX_BUTTON_X:         std::cout << "Button X pressed\n"; break;
                    case XBOX_BUTTON_Y:         std::cout << "Button Y pressed\n"; break;

                    // shoulder buttons
                    case XBOX_BUTTON_LB:        std::cout << "Left Bumper (LB) pressed\n"; break;
                    case XBOX_BUTTON_RB:        std::cout << "Right Bumper (RB) pressed\n"; break;

                    // stick clicks
                    case XBOX_BUTTON_LS:        std::cout << "Left Stick Click (LS) pressed\n"; break;
                    case XBOX_BUTTON_RS:        std::cout << "Right Stick Click (RS) pressed\n"; break;

                    // menu / Options
                    case XBOX_BUTTON_BACK:      std::cout << "Back button pressed\n"; break;
                    case XBOX_BUTTON_START:     std::cout << "Start button pressed\n"; break;
                    case XBOX_BUTTON_HOME:      std::cout << "Home/Guide button pressed\n"; break;

                    default:
                        std::cout << "Unmapped input\n";
                        break;
                }
            }
        }
    }

    close(fd);
    return 0;
}
