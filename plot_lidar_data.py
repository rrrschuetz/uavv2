import socket
import struct
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def connect_lidar(ip, port=8089):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
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
    if sync1 != 0xA or sync2 != 0x5:
        raise ValueError(f"Invalid sync bytes: sync1={sync1:#04x}, sync2={sync2:#04x}")

    checksum_received = ((packet[1] & 0x0F) << 4) | (packet[0] & 0x0F)
    checksum_computed = 0
    for byte in packet[2:]:
        checksum_computed ^= byte

    if checksum_received != checksum_computed:
        raise ValueError(f"Checksum validation failed: received={checksum_received:#04x}, computed={checksum_computed:#04x}")

    start_angle_q6 = ((packet[2] & 0xFF) | ((packet[3] & 0xFF) << 8)) >> 1
    start_angle = start_angle_q6 / 64.0

    angle_diff_q3 = ((packet[3] & 0x01) << 8) | (packet[4] & 0xFF)
    angle_diff = angle_diff_q3 / 8.0

    distances = []
    angles = []

    for i in range(40):
        index = 4 + i * 2  # Corrected to index + 4
        if index + 1 >= len(packet):
            raise ValueError(f"Packet is too short for expected data: index {index}")
        distance = (packet[index] & 0xFF) | ((packet[index + 1] & 0xFF) << 8)
        distances.append(distance)
        angle = start_angle + i * angle_diff
        if angle >= 360.0:
            angle -= 360.0
        angles.append(angle)

    return {
        "start_angle": start_angle,
        "distances": distances,
        "angles": angles
    }

def initialize(sock):
    pass

def main():
    IP_ADDRESS = '192.168.11.2'
    PORT = 8089

    fig, ax = plt.subplots()
    scatter = ax.scatter([], [], s=1)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    all_x_coords = []
    all_y_coords = []

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
                angle_rad = np.radians(angle)
                x = distance * 0.001 * np.cos(angle_rad)  # Convert mm to meters
                y = distance * 0.001 * np.sin(angle_rad)
                x_coords.append(x)
                y_coords.append(y)

            all_x_coords.extend(x_coords)
            all_y_coords.extend(y_coords)
            scatter.set_offsets(np.c_[all_x_coords, all_y_coords])
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
