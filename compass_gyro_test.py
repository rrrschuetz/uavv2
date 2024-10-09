import time
import math
import numpy as np
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
TIMEOUT = 0.5
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)

# Kalman filter parameters
Q = 0.1  # Process noise covariance
R = 1.0  # Measurement noise covariance
P = 1.0  # Estimation error covariance
heading_estimate = 0.0  # Initial heading estimate

# Kalman filter function
def kalman_filter(gyro_heading_change, mag_heading, heading_estimate, P):
    # Prediction step
    P = P + Q  # Update process error covariance with process noise
    heading_predict = heading_estimate + gyro_heading_change  # Predict the new heading

    # Measurement update step
    K = P / (P + R)  # Calculate Kalman gain
    heading_estimate = heading_predict + K * (mag_heading - heading_predict)  # Update estimate with measurement
    P = (1 - K) * P  # Update error covariance

    return heading_estimate, P

# Get magnetometer heading
def vector_2_degrees(x, y):
    angle = math.degrees(math.atan2(y, x))
    if angle < 0:
        angle += 360
    return angle

def get_magnetometer_heading():
    mag_x, mag_y, _ = qmc.magnetic
    return vector_2_degrees(mag_x, mag_y)

# Get gyroscope yaw data from WT61
def get_gyro_yaw():
    buff = bytearray()
    yaw = None

    if ser.in_waiting:
        data = ser.read(ser.in_waiting)
        for byte in data:
            buff.append(byte)
            if len(buff) >= 11:  # Ensure we have enough data for a full packet (11 bytes)
                if buff[0] == 0x55 and buff[1] == 0x53:  # Check for correct start of packet and type
                    try:
                        _, _, _, yaw = struct.unpack('<hhh', buff[2:8])
                        yaw = yaw / 32768.0 * 180  # Convert to degrees
                    except struct.error as e:
                        print(f"Unpacking error: {e}")
                    finally:
                        buff = buff[11:]  # Remove processed packet from the buffer
    return yaw

# Main loop for Kalman filtering
def main():
    global heading_estimate, P
    last_time = time.time()

    while True:
        # Get the magnetometer heading (absolute heading)
        mag_heading = get_magnetometer_heading()

        # Get the gyroscope yaw rate (relative heading change)
        gyro_yaw = get_gyro_yaw()

        # Calculate time difference for integrating gyroscope data
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        if gyro_yaw is not None:
            # Integrate gyroscope yaw to get relative change in heading
            gyro_heading_change = gyro_yaw * dt

            # Apply Kalman filter to fuse magnetometer and gyroscope data
            heading_estimate, P = kalman_filter(gyro_heading_change, mag_heading, heading_estimate, P)

            # Print the stable, filtered heading
            print(f"Kalman Filtered Heading: {heading_estimate:.2f} degrees")

        # Update at 10Hz
        time.sleep(0.1)

if __name__ == "__main__":
    main()
