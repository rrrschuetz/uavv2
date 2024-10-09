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

# Initialize serial connection for WT61 (gyroscope and accelerometer)
SERIAL_PORT = "/dev/ttyAMA0"  # Adjust if needed
BAUD_RATE = 115200
TIMEOUT = 0.5
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)

# Kalman filter parameters
Q = 0.1  # Process noise covariance
R = 1.0  # Measurement noise covariance
P = 1.0  # Estimation error covariance
heading_estimate = 0.0  # Initial heading estimate
pitch_estimate = 0.0  # Initial pitch estimate
roll_estimate = 0.0  # Initial roll estimate

# Kalman filter function for yaw
def kalman_filter(gyro_heading_change, mag_heading, heading_estimate, P):
    P = P + Q  # Update process error covariance with process noise
    heading_predict = heading_estimate + gyro_heading_change  # Predict the new heading
    K = P / (P + R)  # Calculate Kalman gain
    heading_estimate = heading_predict + K * (mag_heading - heading_predict)  # Update estimate
    P = (1 - K) * P  # Update error covariance
    return heading_estimate, P

# Get magnetometer heading
def vector_2_degrees(x, y):
    angle = math.degrees(math.atan2(y, x))
    if angle < 0:
        angle += 360
    return angle

def get_magnetometer_heading():
    mag_x, mag_y, mag_z = qmc.magnetic
    return mag_x, mag_y, mag_z

# Tilt compensation for magnetometer using pitch and roll
def tilt_compensate(mag_x, mag_y, mag_z, pitch, roll):
    # Apply tilt compensation to magnetometer readings using pitch and roll
    mag_x_comp = mag_x * math.cos(pitch) + mag_z * math.sin(pitch)
    mag_y_comp = mag_x * math.sin(roll) * math.sin(pitch) + mag_y * math.cos(roll) - mag_z * math.sin(roll) * math.cos(pitch)
    return mag_x_comp, mag_y_comp

# Get gyroscope and accelerometer data from WT61
def get_gyro_accel_data():
    buff = bytearray()
    accel = (None, None, None)
    gyro = (None, None, None)

    if ser.in_waiting:
        data = ser.read(ser.in_waiting)
        for byte in data:
            buff.append(byte)
            if len(buff) >= 11:  # Ensure we have enough data for a full packet (11 bytes)
                if buff[0] == 0x55:
                    try:
                        data_type = buff[1]
                        values = struct.unpack('<hhh', buff[2:8])
                        if data_type == 0x51:  # Accelerometer data
                            accel = [v / 32768.0 * 16 for v in values]  # Convert to G
                        elif data_type == 0x53:  # Gyroscope angle data (roll, pitch, yaw)
                            gyro = [v / 32768.0 * 180 for v in values]  # Convert to degrees
                    except struct.error as e:
                        print(f"Unpacking error: {e}")
                    finally:
                        buff = buff[11:]  # Remove processed packet from the buffer

    return gyro, accel

# Compute pitch and roll from accelerometer data
def compute_pitch_roll(accel_x, accel_y, accel_z):
    pitch = math.atan2(accel_y, math.sqrt(accel_x**2 + accel_z**2))
    roll = math.atan2(-accel_x, accel_z)
    return pitch, roll

# Main loop for Kalman filtering
def main():
    global heading_estimate, P, pitch_estimate, roll_estimate
    last_time = time.time()

    while True:
        # Get gyroscope and accelerometer data
        gyro_data, accel_data = get_gyro_accel_data()

        if accel_data != (None, None, None):
            accel_x, accel_y, accel_z = accel_data

            # Calculate pitch and roll from accelerometer data
            pitch_estimate, roll_estimate = compute_pitch_roll(accel_x, accel_y, accel_z)

        if gyro_data != (None, None, None):
            roll, pitch, yaw = gyro_data

        # Get the magnetometer heading (absolute heading)
        mag_x, mag_y, mag_z = get_magnetometer_heading()

        # Tilt compensate the magnetometer data
        mag_x_comp, mag_y_comp = tilt_compensate(mag_x, mag_y, mag_z, pitch_estimate, roll_estimate)

        # Calculate the magnetometer heading
        mag_heading = vector_2_degrees(mag_x_comp, mag_y_comp)

        # Calculate time difference for integrating gyroscope data
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        if gyro_data != (None, None, None):
            # Integrate gyroscope yaw (gyro_z) to get relative change in heading
            gyro_heading_change = yaw * dt

            # Apply Kalman filter to fuse magnetometer and gyroscope data
            heading_estimate, P = kalman_filter(gyro_heading_change, mag_heading, heading_estimate, P)

            # Print the stable, filtered heading
            print(f"Kalman Filtered Heading: {heading_estimate:.2f} degrees")

        # Update at 10Hz
        time.sleep(0.1)

if __name__ == "__main__":
    main()
