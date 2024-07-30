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
    # Start standard scan mode
    sock.send(b'\xA5\x20')
    response = receive_full_data(sock, 10)
    print(f"Start scan response: {response}")

def stop_scan(sock):
    sock.send(b'\xA5\x25')
    time.sleep(0.1)

def decode_standard_mode_packet(packet):
    if len(packet) < 5:
        raise ValueError(f"Invalid packet length: {len(packet)}")

    angle_q6 = (packet[2] << 8) | (packet[1] >> 1)
    angle = angle_q6 / 64
    distance = ((packet[4] << 8) | packet[3])

    return angle, distance

def initialize(sock):
    print("Initializing LIDAR...")
    health = get_health(sock)
    print("LIDAR health:", health)
    if health[0] != 0:  # Check if health status is good
        raise Exception("LIDAR health is not good")

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
            data = receive_full_data(sock, 5)
            angle, distance = decode_standard_mode_packet(data)
            x_coords = []
            y_coords = []
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
