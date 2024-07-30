import socket
import struct
import time

def connect_lidar(ip, port=8089):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Use UDP
    sock.connect((ip, port))
    return sock

def receive_sync_bytes(sock, timeout=5):
    sock.settimeout(timeout)
    while True:
        try:
            byte = sock.recv(1)
            if byte == b'\xA5':
                next_byte = sock.recv(1)
                if next_byte == b'\x5A':
                    return b'\xA5\x5A'
        except socket.timeout:
            raise TimeoutError("Sync bytes not received within timeout period")

def receive_full_data(sock, expected_length):
    data = b''
    while len(data) < expected_length:
        packet, _ = sock.recvfrom(expected_length - len(data))
        data += packet
    return data

def get_health(sock):
    sock.send(b'\xA5\x52')
    response = receive_full_data(sock, 10)
    print(f"Received health data: {response}")
    status, error_code = struct.unpack('<BH', response[3:6])
    return status, error_code

def get_info(sock):
    sock.send(b'\xA5\x50')
    response = receive_full_data(sock, 27)
    print(f"Received info data: {response}")
    model, firmware_minor, firmware_major, hardware, serialnum = struct.unpack('<BBBB16s', response[7:])
    serialnum_str = serialnum[::-1].hex()
    return model, firmware_minor, firmware_major, hardware, serialnum_str

def start_scan(sock):
    sock.send(b'\xA5\x82\x05\x00\x00\x00\x00\x00\x22')
    response = receive_full_data(sock, 7)
    print(f"Start scan response: {response}")

def stop_scan(sock):
    sock.send(b'\xA5\x25')
    time.sleep(0.1)

def decode_dense_mode_packet(packet):
    if len(packet) < 84:  # Minimum length for a dense packet
        raise ValueError("Packet too short to be valid")

    sync1 = packet[0]
    sync2 = packet[1]
    #if sync1 != 0xA5 or sync2 != 0x5A:
    #    raise ValueError(f"Invalid sync bytes: sync1={sync1:#04x}, sync2={sync2:#04x}")

    checksum_received = packet[2]
    checksum_computed = 0
    for byte in packet[3:]:
        checksum_computed ^= byte

    #if checksum_received != checksum_computed:
    #    raise ValueError(f"Checksum validation failed: received={checksum_received:#04x}, computed={checksum_computed:#04x}")

    start_angle_q6 = ((packet[3] & 0xFF) << 8) | (packet[4] & 0xFF)
    start_angle = start_angle_q6 / 64.0

    distances = []
    for i in range(40):
        index = 5 + i * 2
        if index + 1 >= len(packet):
            raise ValueError(f"Packet is too short for expected data: index {index}")
        distance = (packet[index] & 0xFF) | ((packet[index + 1] & 0xFF) << 8)
        distances.append(distance)

    angles = []
    angle_diff = 360 / 40
    for i in range(40):
        angle = start_angle + (i * angle_diff)
        if angle >= 360:
            angle -= 360
        angles.append(angle)

    return {
        "start_angle": start_angle,
        "distances": distances,
        "angles": angles
    }

def initialize(sock):
    pass

def main():
    IP_ADDRESS = '192.168.11.2'  # Replace with your LIDAR's IP address
    PORT = 8089

    sock = connect_lidar(IP_ADDRESS, PORT)
    try:
        print('Initializing LIDAR...')
        initialize(sock)
        print('Initialization complete.')

        print('Getting LIDAR info...')
        info = get_info(sock)
        print('LIDAR Info:', info)

        print('Getting LIDAR health...')
        health = get_health(sock)
        print('LIDAR Health:', health)

        print('Starting scan...')
        start_scan(sock)
        try:
            while True:
                data = receive_full_data(sock, 88)
                print(f"Received data: {data}")
                decoded_data = decode_dense_mode_packet(data)
                print(f"Start Angle: {decoded_data['start_angle']}")
                print(f"Distances: {decoded_data['distances']}")
                print(f"Angles: {decoded_data['angles']}")
        except KeyboardInterrupt:
            pass
        finally:
            print('Stopping scan...')
            stop_scan(sock)
    finally:
        sock.close()

if __name__ == '__main__':
    main()
