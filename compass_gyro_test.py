import time
from math import atan2, degrees
import board
import qmc5883l as qmc5883
import serial
import struct

# Initialize I2C for QMC5883L (magnetometer)
i2c = board.I2C()
qmc = qmc5883.QMC5883L(i2c)
qmc.output_data_rate = (qmc5883.OUTPUT_DATA_RATE_200)

# Initialize serial connection for WT61 (gyroscope)
SERIAL_PORT = "/dev/ttyAMA0"  # Adjust if needed
BAUD_RATE = 115200
TIMEOUT = 0.1

# Create a serial object for the gyroscope
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)

# Rolling average setup
ROLLING_WINDOW_SIZE = 10
heading_buffer = []

# Get magnetometer heading
def vector_2_degrees(x, y):
    angle = degrees(atan2(y, x))
    if angle < 0:
        angle += 360
    return angle

def get_magnetometer_heading():
    mag_x, mag_y, _ = qmc.magnetic
    return vector_2_degrees(mag_x, mag_y)

# Get gyroscope yaw data from WT61
def get_gyro_yaw():
    buff = bytearray()
    data_type = None
    yaw = None

    if ser.in_waiting:
        data = ser.read(ser.in_waiting)
        for byte in data:
            buff.append(byte)
            if len(buff) >= 11:
                if buff[0] == 0x55 and buff[1] == 0x53:  # Check for angle data packet
                    _, _, _, yaw = struct.unpack('<hhh', buff[2:8])
                    yaw = yaw / 32768.0 * 180  # Convert to degrees
                buff = buff[11:]  # Clear buffer for next packet

    return yaw

# Rolling average for smoothing
def rolling_average(heading, buffer, window_size):
    buffer.append(heading)
    if len(buffer) > window_size:
        buffer.pop(0)
    return sum(buffer) / len(buffer)

# Main loop to combine magnetometer and gyroscope data
def main():
    current_heading = get_magnetometer_heading()  # Initialize with magnetometer
    last_time = time.time()

    while True:
        # Get the magnetometer heading
        mag_heading = get_magnetometer_heading()

        # Get the gyroscope yaw rate (change in heading)
        gyro_yaw = get_gyro_yaw()

        # Update the heading based on gyroscope
        if gyro_yaw is not None:
            elapsed_time = time.time() - last_time
            current_heading += gyro_yaw * elapsed_time  # Update using gyro data
            current_heading %= 360  # Keep heading in [0, 360]
            last_time = time.time()

        # Every 1 second, use the magnetometer to correct for drift
        if time.time() - last_time > 1:
            current_heading = mag_heading

        # Apply rolling average for stability
        stable_heading = rolling_average(current_heading, heading_buffer, ROLLING_WINDOW_SIZE)

        # Print the stable heading
        print(f"Stable Heading: {stable_heading:.2f} degrees")

        # Update at 10Hz
        time.sleep(0.1)

if __name__ == "__main__":
    main()
