# xbox_controls.py
import sys, termios, tty

XBOX_GAS_TRIGGER       = 0x09
XBOX_BRAKE_TRIGGER     = 0x0A

XBOX_LEFT_STICK_X      = 0x00
XBOX_LEFT_STICK_Y      = 0x01
XBOX_RIGHT_STICK_X     = 0x05
XBOX_RIGHT_STICK_Y     = 0x02

XBOX_DPAD_X            = 0x10
XBOX_DPAD_Y            = 0x11

XBOX_BUTTON_A          = 0x130
XBOX_BUTTON_B          = 0x131
XBOX_BUTTON_X          = 0x133
XBOX_BUTTON_Y          = 0x134

XBOX_BUTTON_LB         = 0x136
XBOX_BUTTON_RB         = 0x137

XBOX_BUTTON_LS         = 0x13D
XBOX_BUTTON_RS         = 0x13E

XBOX_BUTTON_BACK       = 0x13A
XBOX_BUTTON_START      = 0x13B
XBOX_BUTTON_HOME       = 0x13C

DEFAULT_SPEED          = 5200
DEFAULT_STEERING       = 4990

STEERING_PIN           = 0
SPEED_PIN              = 8

DEFAULT_SPEED_INCREMENT = 420
DEFAULT_STEERING_INCREMENT = 100

DEFAULT_FULL_RIGHT = 6400
DEFAULT_FULL_LEFT = 3680

DEFAULT_REVERSE_FULL_SPEED = 3680
DEFAULT_FORWARD_FULL_SPEED = 6400

DEFAULT_REVERSE_FULL_SPEED_LIMITED = 3680
DEFAULT_FORWARD_FULL_SPEED_LIMITED = 6400

SPEED_STEP = 75
OFFSET_FROM_STOP_TO_START_MOVING = 210  # offset to overcome motor deadzone

def get_key():
    tty.setraw(sys.stdin.fileno())
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, termios.tcgetattr(sys.stdin))
    return key

def map_steering_value(angle, deadzone=0.0):
    """
    Map angle (-3 to 3) to steering PWM, with deadzone and special case for 755.
    """

    if angle == 755:
        return 50
    if abs(angle) <= deadzone:
        return 50
    
    if angle < -3:
        angle = 0
    elif angle > 3:
        angle = 0
    
    # Map -3..3 to -3660..6400
    shifted = angle + 3  # now in [0, 6]                 x       y
    percentage = shifted * 100 / 6  # 0..100%         100     6 

    return percentage


def remap_stick_value_to_0_100_range(raw_value, deadzone=3000):
    """
    Maps raw value-axis controller input (0 to 65535) to a 0 to 100 scale, with deadzone handling.

    Parameters:
        raw_value (int): Raw value input from the controller (0 to 65535)
        deadzone (int): Optional deadzone to ignore small movements near the center

    Returns:
        int: Value between 0 and 100
    """
    center = 32768
    max_offset = 32767  # max deviation from center

    offset = raw_value - center

    # apply deadzone
    if abs(offset) < deadzone:
        return 50 

    normalized = offset / max_offset
    # map to [0, 100]
    mapped = int((normalized + 1) * 50)
    
    return max(0, min(100, mapped))