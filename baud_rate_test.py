import serial

# List of common baud rates to try
baud_rates = [9600, 19200, 38400, 57600, 115200, 230400]

SERIAL_PORT = "/dev/ttyAMA0"  # Update this if you're using a different serial port


def test_baud_rates():
    for baud_rate in baud_rates:
        try:
            with serial.Serial(SERIAL_PORT, baud_rate, timeout=1) as ser:
                print(f"Testing baud rate: {baud_rate}")
                ser.reset_input_buffer()  # Clear any old data from the input buffer

                # Read some data to see if it makes sense
                data = ser.read(20)  # Read 20 bytes
                if data:
                    print(f"Received data at baud rate {baud_rate}: {data.hex()}")
                    # Look for readable characters (assuming ASCII output is expected)
                    if any(32 <= byte <= 126 for byte in data):
                        print(f"Likely match at baud rate {baud_rate}: {data.decode('utf-8', errors='ignore')}")
        except Exception as e:
            print(f"Error testing baud rate {baud_rate}: {e}")


if __name__ == "__main__":
    test_baud_rates()
