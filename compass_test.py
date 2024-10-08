import smbus
import time
import math

# Define the I2C bus number (1 on most Raspberry Pi models)
I2C_BUS = 1
SENSOR_ADDRESS = 0x0D

bus = smbus.SMBus(I2C_BUS)

# Register definitions for QMC5883L
REG_CONTROL_1 = 0x09
REG_DATA_X_LSB = 0x00

bus.write_byte_data(SENSOR_ADDRESS, REG_CONTROL_1, 0x1D)

# Calibration values (set these manually after rotating the sensor in all directions)
# Replace these with actual min/max values after calibration
X_min = -500
X_max = 500
Y_min = -500
Y_max = 500
Z_min = -500
Z_max = 500


def read_raw_data(register):
    low_byte = bus.read_byte_data(SENSOR_ADDRESS, register)
    high_byte = bus.read_byte_data(SENSOR_ADDRESS, register + 1)
    value = (high_byte << 8) | low_byte
    if value > 32767:
        value -= 65536
    return value


def read_magnetometer_data():
    x = read_raw_data(REG_DATA_X_LSB)
    y = read_raw_data(REG_DATA_X_LSB + 2)
    z = read_raw_data(REG_DATA_X_LSB + 4)
    return x, y, z


def calibrate(x, y, z):
    # Apply calibration to raw data to adjust for hard/soft iron distortions
    x_offset = (X_max + X_min) / 2
    y_offset = (Y_max + Y_min) / 2
    z_offset = (Z_max + Z_min) / 2

    x_range = (X_max - X_min) / 2
    y_range = (Y_max - Y_min) / 2
    z_range = (Z_max - Z_min) / 2

    x_cal = (x - x_offset) / x_range
    y_cal = (y - y_offset) / y_range
    z_cal = (z - z_offset) / z_range

    return x_cal, y_cal, z_cal


def calculate_heading(x, y):
    heading_radians = math.atan2(y, x)
    heading_degrees = math.degrees(heading_radians)
    if heading_degrees < 0:
        heading_degrees += 360
    return heading_degrees


try:
    while True:
        x, y, z = read_magnetometer_data()

        # Calibrate magnetometer data
        x_cal, y_cal, z_cal = calibrate(x, y, z)

        # Calculate heading from calibrated data
        heading = calculate_heading(x_cal, y_cal)

        print(f"Heading: {heading:.2f} degrees")

        time.sleep(1)
except KeyboardInterrupt:
    print("Measurement stopped by user")
