#ifndef XBOX_CONTROLS_H
#define XBOX_CONTROLS_H

// Triggers (Analog Axes)
#define XBOX_GAS_TRIGGER       0x09  // ABS_Z    -> Right Trigger (RT)
#define XBOX_BRAKE_TRIGGER     0x0a  // ABS_RZ   -> Left Trigger (LT)

// Joysticks (Analog Axes)
#define XBOX_LEFT_STICK_X      0x01  // ABS_X    -> Left Stick Left/Right
#define XBOX_LEFT_STICK_Y      0x00  // ABS_Y    -> Left Stick Up/Down
#define XBOX_RIGHT_STICK_X     0x02  // ABS_RX   -> Right Stick Left/Right
#define XBOX_RIGHT_STICK_Y     0x05  // ABS_RY   -> Right Stick Up/Down

// D-Pad (Digital Buttons via ABS Hat)
#define XBOX_DPAD_X            0x10  // ABS_HAT0X
#define XBOX_DPAD_Y            0x11  // ABS_HAT0Y

// Face Buttons (Digital Buttons)
#define XBOX_BUTTON_A          0x130 // BTN_SOUTH
#define XBOX_BUTTON_B          0x131 // BTN_EAST
#define XBOX_BUTTON_X          0x133 // BTN_WEST
#define XBOX_BUTTON_Y          0x134 // BTN_NORTH

// Shoulder Buttons (Digital)
#define XBOX_BUTTON_LB         0x136 // BTN_TL
#define XBOX_BUTTON_RB         0x137 // BTN_TR

// Stick Clicks (Digital)
#define XBOX_BUTTON_LS         0x13D // BTN_THUMBL
#define XBOX_BUTTON_RS         0x13E // BTN_THUMBR


// Menu / Options / Guide Buttons
#define XBOX_BUTTON_BACK       0x13A // BTN_SELECT
#define XBOX_BUTTON_START      0x13B // BTN_START
#define XBOX_BUTTON_HOME       0x13C // BTN_MODE

#endif // XBOX_CONTROLS_H

