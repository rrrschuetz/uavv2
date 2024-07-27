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


try:
    # Arm the ESC by setting it to minimum speed
    set_motor_speed(0, 0)
    time.sleep(1)

    while True:
        # Example usage: gradually increase speed from 0% to 100%
        for speed in range(0, 101, 10):
            set_motor_speed(0, speed)
            print(f"Speed: {speed}%")
            time.sleep(1)

        # Gradually decrease speed from 100% to 0%
        for speed in range(100, -1, -10):
            set_motor_speed(0, speed)
            print(f"Speed: {speed}%")
            time.sleep(1)
except KeyboardInterrupt:
    # Stop the motor on exit
    set_motor_speed(0, 0)
    print("Program stopped by User")
finally:
    pca.deinit()
