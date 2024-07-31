import time
from adafruit_pca9685 import PCA9685
from board import SCL, SDA
import busio

# Initialize the I2C bus
i2c = busio.I2C(SCL, SDA)

# Create a simple PCA9685 class instance
pca = PCA9685(i2c)
pca.frequency = 50  # ESCs typically use a 50Hz frequency


def set_motor_speed(channel, speed):
    """
    Sets the speed of the motor connected to the specified channel.

    :param channel: The channel number on the PCA9685 where the ESC is connected (0-15).
    :param speed: The desired speed of the motor (0-100%).
    """
    # Convert the speed to the corresponding pulse width
    pulse_min = 150  # Pulse width for 0% speed
    pulse_max = 600  # Pulse width for 100% speed
    pulse_width = pulse_min + (speed / 100.0) * (pulse_max - pulse_min)

    # Set the pulse width for the specified channel
    pca.channels[channel].duty_cycle = int(pulse_width / 4096 * 0xFFFF)


def calibrate_esc(channel):
    """
    Calibrates the ESC by setting the maximum and minimum throttle positions.

    :param channel: The channel number on the PCA9685 where the ESC is connected (0-15).
    """
    print("Calibrating ESC...")

    # Set to maximum throttle
    print("Setting to maximum throttle...")
    set_motor_speed(channel, 100)
    time.sleep(2)

    # Set to minimum throttle
    print("Setting to minimum throttle...")
    set_motor_speed(channel, 0)
    time.sleep(2)

    print("Calibration complete")


def arm_esc(channel):
    """
    Arms the ESC by setting it to minimum throttle for a short period.

    :param channel: The channel number on the PCA9685 where the ESC is connected (0-15).
    """
    print("Arming ESC...")
    set_motor_speed(channel, 0)
    time.sleep(1)
    print("ESC armed")


try:
    # Calibrate the ESC
    calibrate_esc(1)

    # Arm the ESC
    arm_esc(1)

    while True:
        # Example usage: gradually increase speed from 0% to 100%
        for speed in range(0, 101, 10):
            set_motor_speed(1, speed)
            print(f"Speed: {speed}%")
            time.sleep(1)

        # Gradually decrease speed from 100% to 0%
        for speed in range(100, -1, -10):
            set_motor_speed(1,speed)
            print(f"Speed: {speed}%")
            time.sleep(1)
except KeyboardInterrupt:
    # Stop the motor on exit
    set_motor_speed(1,0)
    print("Program stopped by User")
finally:
    pca.deinit()
