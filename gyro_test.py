import serial
import struct
import time

# Configuration for the WT61 gyroscope
SERIAL_PORT = "/dev/serial0"  # or "/dev/ttyS0"
BAUD_RATE = 115200  # WT61 usually works with 115200 baud rate
TIMEOUT = 1  # Serial timeout in seconds


# Function to parse data from the WT61 sensor packet
def parse_wt61_data(data):
    try:
        if len(data) == 11 and data[0] == 0x55:
            data_type = data[1]
            values = struct.unpack('<hhh', data[2:8])  # Convert bytes to short integers
            value1, value2, value3 = values
            return data_type, value1 / 32768.0 * 2000, value2 / 32768.0 * 2000, value3 / 32768.0 * 2000
    except Exception as e:
        print(f"Error parsing data: {e}")
    return None, None, None, None


# Function to read from serial port
def read_gyro_data():
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT) as ser:
            while True:
                # Attempt to read a full frame
                data = ser.read(11)  # Read 11 bytes, which is a typical frame size
                if len(data) == 11:
                    data_type, val1, val2, val3 = parse_wt61_data(data)
                    if data_type == 0x52:  # 0x52 indicates it's angular velocity data
                        print(f"Gyro Data -> X: {val1:.2f}, Y: {val2:.2f}, Z: {val3:.2f} (deg/s)")
                    elif data_type == 0x53:  # 0x53 indicates it's angle data
                        print(f"Angle Data -> Roll: {val1:.2f}, Pitch: {val2:.2f}, Yaw: {val3:.2f} (degrees)")
                    else:
                        print("Unknown data type received.")

                time.sleep(0.1)  # Small delay to avoid overwhelming the CPU

    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except KeyboardInterrupt:
        print("Stopping data read.")


# Start reading data from WT61 gyro
if __name__ == "__main__":
    read_gyro_data()
