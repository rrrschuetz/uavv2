import socket
import struct
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def connect_lidar(ip, port=8089):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Use UDP
    sock.connect((ip, port))
    return sock

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
    response = receive_full_data(sock, 10)
    print(f"Start scan response: {response}")

def stop_scan(sock):
    sock.send(b'\xA5\x25')
    time.sleep(0.1)

def decode_dense_mode_packet(packet):
    if len(packet) != 84:
        raise ValueError(f"Invalid packet length: {len(packet)}")

    sync1 = (packet[0] & 0xF0) >> 4
    sync2 = (packet[1] & 0xF0) >> 4
    if sync1 != 0xA and sync2 != 0x5:
        raise ValueError(f"Invalid sync bytes: sync1={sync1:#04x}, sync2={sync2:#04x}")

    checksum_received = ((packet[1] & 0x0F) << 4) | (packet[0] & 0x0F)
    checksum_computed = 0
    for byte in packet[2:]:
        checksum_computed ^= byte

    if checksum_received != checksum_computed:
        raise ValueError(f"Checksum validation failed: received={checksum_received:#04x}, computed={checksum_computed:#04x}")

    start_angle_q6 = ((packet[2] & 0xFF) << 8) | (packet[3] & 0xFF)
    start_angle = start_angle_q6 / 64.0

    distances = []
    for i in range(40):
        index = 4 + i * 2
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

    fig, ax = plt.subplots()
    scatter = ax.scatter([], [])
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

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

        def update(frame):
            data = receive_full_data(sock, 84)
            decoded_data = decode_dense_mode_packet(data)
            x_coords = []
            y_coords = []
            for angle, distance in zip(decoded_data['angles'], decoded_data['distances']):
                angle_rad = angle * (3.14159 / 180.0)
                x = distance * 0.001 * np.cos(angle_rad)  # Convert mm to meters
                y = distance * 0.001 * np.sin(angle_rad)
                x_coords.append(x)
                y_coords.append(y)
            scatter.set_offsets(np.c_[x_coords, y_coords])
            return scatter,

        ani = FuncAnimation(fig, update, interval=100)
        plt.show()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            print('Stopping scan...')
            stop_scan(sock)
    finally:
        sock.close()

if __name__ == '__main__':
    main()
