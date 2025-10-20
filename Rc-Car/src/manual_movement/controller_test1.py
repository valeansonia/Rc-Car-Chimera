import struct
import os
import math

from xbox_controls2 import *

device = "/dev/input/event13"

try:
    fd = os.open(device, os.O_RDONLY)
except OSError:
    print(f"Failed to open {device}")
    exit(1)

print("LOGGING: Listening for Xbox controller input...")

def remap_x_to_0_100(x_raw, deadzone=3000):
    """
    Maps raw X-axis controller input (0 to 65535) to a 0 to 100 scale, with deadzone handling.

    Parameters:
        x_raw (int): Raw X-axis input from the controller (0 to 65535)
        deadzone (int): Optional deadzone to ignore small movements near the center

    Returns:
        int: Value between 0 and 100
    """
    center = 32768
    max_offset = 32767  # Max deviation from center

    offset = x_raw - center

    # Apply deadzone
    if abs(offset) < deadzone:
        return 50  # Neutral position (center)

    # Normalize to range [-1, 1]
    normalized = offset / max_offset
    # Map to [0, 100]
    mapped = int((normalized + 1) * 50)
    
    return max(0, min(100, mapped))



FORMAT = 'llHHI'
EVENT_SIZE = struct.calcsize(FORMAT)

while True:
    data = os.read(fd, EVENT_SIZE)
    if data:
        sec, usec, type_, code, value = struct.unpack(FORMAT, data)

        if (type_ == 1 or type_ == 3) and value != 0:
            if code == XBOX_LEFT_STICK_Y:
                print(f"LOGGING: Left Stick X moved: {remap_x_to_0_100(value)}")
                # print(f"Code: 0x{code:02X} | Value: {value} → ", end='')

            if code == XBOX_GAS_TRIGGER:
                print("LOGGING: Gas Trigger (RT) pressed")
            elif code == XBOX_BRAKE_TRIGGER:
                print("LOGGING: Brake Trigger (LT) pressed")
            elif code == XBOX_LEFT_STICK_X:
                print("LOGGING: Left Stick X moved")
            elif code == XBOX_LEFT_STICK_Y:
                print("LOGGING: Left Stick Y moved")
            elif code == XBOX_RIGHT_STICK_X:
                print("LOGGING: Right Stick X moved")
            elif code == XBOX_RIGHT_STICK_Y:
                print("LOGGING: Right Stick Y moved")
            elif code == XBOX_DPAD_X:
                print("LOGGING: D-Pad Left/Right")
            elif code == XBOX_DPAD_Y:
                print("LOGGING: D-Pad Up/Down")
            elif code == XBOX_BUTTON_A:
                print("LOGGING: Button A pressed")
            elif code == XBOX_BUTTON_B:
                print("LOGGING: Button B pressed")
            elif code == XBOX_BUTTON_X:
                print("LOGGING: Button X pressed")
            elif code == XBOX_BUTTON_Y:
                print("LOGGING: Button Y pressed")
            elif code == XBOX_BUTTON_LB:
                print("LOGGING: Left Bumper (LB) pressed")
            elif code == XBOX_BUTTON_RB:
                print("LOGGING: Right Bumper (RB) pressed")
            elif code == XBOX_BUTTON_LS:
                print("LOGGING: Left Stick Click (LS) pressed")
            elif code == XBOX_BUTTON_RS:
                print("LOGGING: Right Stick Click (RS) pressed")
            elif code == XBOX_BUTTON_BACK:
                print("LOGGING: Back button pressed")
            elif code == XBOX_BUTTON_START:
                print("LOGGING: Start button pressed")
            elif code == XBOX_BUTTON_HOME:
                print("LOGGING: Home/Guide button pressed")
            else:
                print("LOGGING: Unmapped input")
