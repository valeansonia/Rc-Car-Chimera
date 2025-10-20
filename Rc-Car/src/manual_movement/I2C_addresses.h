#ifndef PWM_CONTROL_H
#define PWM_CONTROL_H

#include <cstdint>

bool initPCA9685();
void setPWM(uint8_t channel, uint16_t on, uint16_t off);

#endif