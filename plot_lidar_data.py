import socket
import struct
import time
import numpy as np
import matplotlib.pyplot as plt

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

def decode_dense_mode_packet(packet, old_start_angle=0.0):
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

    start_angle_q6 = (packet[2] | ((packet[3] & 0x7f) << 8))
    start_angle = start_angle_q6 / 64.0
    angle_diff =  start_angle - old_start_angle
    if angle_diff < 0: angle_diff += 360.0

    distances = []
    angles = []

    for i in range(40):
        index = 4 + i * 2  # Corrected to index + 4
        if index + 1 >= len(packet):
            raise ValueError(f"Packet is too short for expected data: index {index}")
        distance = (packet[index] | (packet[index+1] << 8))
        distances.append(distance)
        angle = start_angle + i * angle_diff/40.0
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

    sock = connect_lidar(IP_ADDRESS, PORT)

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

    old_start_angle = 0.0
    all_distances = []
    all_angles = []

    for i in range(2000):
        data = receive_full_data(sock, 84)
        decoded_data = decode_dense_mode_packet(data, old_start_angle)
        old_start_angle = decoded_data['start_angle']
        all_distances.extend(decoded_data['distances'])
        all_angles.extend(decoded_data['angles'])

    print('Stopping scan...')
    stop_scan(sock)
    sock.close()

    # Convert polar coordinates to Cartesian coordinates for plotting
    all_distances = np.array(all_distances) / 1000.0  # Convert to meters if needed
    all_angles = np.radians(all_angles)
    x_coords = all_distances * np.cos(all_angles)
    y_coords = all_distances * np.sin(all_angles)

    # Plotting the data
    fig, ax = plt.subplots()
    ax.scatter(x_coords, y_coords, s=1)
    ax.set_xlim(-0.100, 0.100)
    ax.set_ylim(-0.100, 0.100)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('LIDAR Data')
    plt.show()

if __name__ == '__main__':
    main()
