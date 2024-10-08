import smbus
import time

# Define the I2C bus number (1 on most Raspberry Pi models)
I2C_BUS = 1
# Define the I2C address of the sensor (use i2cdetect to verify, typically 0x0D for QMC5883L)
SENSOR_ADDRESS = 0x0D

# Create I2C bus object
bus = smbus.SMBus(I2C_BUS)

# Register definitions for QMC5883L
REG_CONTROL_1 = 0x09  # Control register for setting modes
REG_DATA_X_LSB = 0x00  # Register where the X-axis LSB is stored

# Initialize the QMC5883L (set continuous measurement mode, 10Hz output rate, 2G range, 512 oversampling)
bus.write_byte_data(SENSOR_ADDRESS, REG_CONTROL_1, 0x1D)


def read_raw_data(register):
    """Read two bytes of data from the specified register."""
    # Read low and high bytes
    low_byte = bus.read_byte_data(SENSOR_ADDRESS, register)
    high_byte = bus.read_byte_data(SENSOR_ADDRESS, register + 1)

    # Combine the two bytes
    value = (high_byte << 8) | low_byte

    # Convert to signed value
    if value > 32767:
        value -= 65536
    return value


def read_magnetometer_data():
    """Read and return X, Y, Z magnetometer data."""
    x = read_raw_data(REG_DATA_X_LSB)
    y = read_raw_data(REG_DATA_X_LSB + 2)
    z = read_raw_data(REG_DATA_X_LSB + 4)

    return x, y, z


try:
    while True:
        # Read magnetometer data
        x, y, z = read_magnetometer_data()

        # Print the raw values
        print(f"X: {x}, Y: {y}, Z: {z}")

        # Wait before reading again
        time.sleep(1)
except KeyboardInterrupt:
    print("Measurement stopped by user")
