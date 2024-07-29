
#sudo apt-get install python3-pip python3-smbus
#pip3 install adafruit-circuitpython-pca9685

from adafruit_pca9685 import PCA9685
from board import SCL, SDA
import busio
import time

# Initialize the I2C bus
i2c = busio.I2C(SCL, SDA)

# Create a simple PCA9685 class instance
pca = PCA9685(i2c)
pca.frequency = 50  # Standard servo frequency


def set_servo_angle(channel, angle):
    """
    Sets the angle of the servo connected to the specified channel.

    :param channel: The channel number on the PCA9685 where the servo is connected (0-15).
    :param angle: The desired angle to set the servo to (0-180 degrees).
    """
    # Convert the angle to the corresponding pulse width
    pulse_min = 260  # Pulse width for 0 degrees
    pulse_max = 380  # Pulse width for 180 degrees
    pulse_width = pulse_min + (angle / 180.0) * (pulse_max - pulse_min)

    # Set the pulse width for the specified channel
    pca.channels[channel].duty_cycle = int(pulse_width / 4096 * 0xFFFF)


try:
    while True:
        # Example usage: sweep the servo back and forth
        for angle in range(0, 181, 10):
            print(f"Setting servo angle to {angle} degrees")
            set_servo_angle(0, angle)
            time.sleep(0.5)
        for angle in range(180, -1, -10):
            print(f"Setting servo angle to {angle} degrees")
            set_servo_angle(0, angle)
            time.sleep(0.5)
except KeyboardInterrupt:
    # Turn off all channels on exit
    pca.deinit()
    print("Program stopped by User")
