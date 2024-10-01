import serial
import struct
import time

# Configuration for WT61 Gyroscope
SERIAL_PORT = "/dev/ttyAMA0"  # or "/dev/ttyS0" if you have mapped accordingly
BAUD_RATE = 115200
TIMEOUT = 0.5  # Set a slightly longer timeout to ensure full packet reads


# Function to parse WT61 data
def parse_wt61_data(data):
    # Check if length is correct
    if len(data) == 11 and data[0] == 0x55:
        if sum(data[0:-1]) & 0xFF == data[-1]:
            data_type = data[1]
            print(f"Data Type: {data_type}")
            values = struct.unpack('<hhh', data[2:8])  # Convert data to three signed short values
            return data_type, values[0] / 32768.0, values[1] / 32768.0, values[2] / 32768.0
        else:
            #print(f"Checksum mismatch: calculated {calculated_checksum:02X}, received {received_checksum:02X}")
            return None, None, None, None
    else:
        #print("Invalid data length")
        return None, None, None, None


# Function to read gyro data from serial
def read_gyro_data():
    buff = bytearray()
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT) as ser:
            ser.reset_input_buffer()  # Clear any old data from the input buffer
            while True:
                if ser.in_waiting:
                    data = ser.read(ser.in_waiting)
                    for byte in data:
                        buff.append(byte)
                        if len(buff) >= 11:  # Assuming each packet length is 11 for now
                            if buff[0] == 0x55:  # Start of packet as per device protocol
                                data_type, val1, val2, val3 = parse_wt61_data(buff[:11])
                                if data_type == 0x53:  # Angle data
                                    print(f"Angle Data -> Roll: {val1 * 180:.2f}, Pitch: {val2 * 180:.2f}, Yaw: {val3 * 180:.2f} (degrees)")
                                elif data_type == 0x51:  # Acceleration data
                                    print(f"Acceleration Data -> X: {val1:.2f}, Y: {val2:.2f}, Z: {val3:.2f}")
                                elif data_type == 0x52:  # Gyroscope data
                                    print(f"Gyro Data -> X: {val1:.2f}, Y: {val2:.2f}, Z: {val3:.2f}")
                            buff = buff[11:]

    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except KeyboardInterrupt:
        print("Stopping data read.")

def initialize_wt61():
    try:
        # Open serial connection to WT61
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT) as ser:
            # Reset the buffer
            ser.reset_input_buffer()

            # Send initialization commands if required by the WT61 (assuming commands are known)
            # For example, sending 0xA5 and 0x54 to set up continuous output (this is just hypothetical)
            init_command = bytes([0xA5, 0x54])
            ser.write(init_command)
            time.sleep(0.1)  # Small delay to allow sensor to process the command

            print("Initialization command sent. Waiting for data...")

            # Read and print the response (if any) from WT61 for debugging purposes
            response = ser.read(20)
            if response:
                print(f"Received response: {response.hex()}")

    except serial.SerialException as e:
        print(f"Serial error: {e}")

# Main function to start reading data
if __name__ == "__main__":
    print("Initializing WT61 gyroscope sensor...")
    initialize_wt61()
    print("Reading data from WT61 gyro sensor...")
    read_gyro_data()
