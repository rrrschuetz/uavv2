import socket
import struct
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Configuration
IP_ADDRESS = '192.168.11.2'
PORT = 8089
LIDAR_LEN = 1620                  # Number of output angles (resolution)                  # Number of output angles (resolution)
ANGLE_CORRECTION = 0.0           # Degrees to offset each angle (if needed)
DISTANCE_CORRECTION = 0.0        # Meters to add to each distance (if needed)


def connect_lidar(ip=IP_ADDRESS, port=PORT):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect((ip, port))
    return sock


def receive_full_data(sock, expected_length, timeout=5):
    sock.settimeout(timeout)
    data = b''
    try:
        while len(data) < expected_length:
            packet, _ = sock.recvfrom(expected_length - len(data))
            data += packet
    except socket.timeout:
        print(f"Timeout ({timeout}s) waiting for {expected_length} bytes.")
        return None
    return data


def get_health(sock):
    sock.send(b'\xA5\x52')
    resp = receive_full_data(sock, 10)
    if not resp:
        raise RuntimeError("Failed to read health status")
    status, error_code = struct.unpack('<BH', resp[3:6])
    return status, error_code


def get_info(sock):
    sock.send(b'\xA5\x50')
    resp = None
    while resp is None:
        resp = receive_full_data(sock, 27)
        if resp is None:
            sock.close()
            sock = connect_lidar()
            start_scan(sock)
    model, fw_min, fw_maj, hw, sn = struct.unpack('<BBBB16s', resp[7:])
    serial = sn[::-1].hex()
    return sock, model, fw_min, fw_maj, hw, serial


def start_scan(sock):
    sock.send(b'\xA5\x82\x05\x00\x00\x00\x00\x00\x22')
    _ = receive_full_data(sock, 10)


def stop_scan(sock):
    sock.send(b'\xA5\x25')
    time.sleep(0.1)


def decode_dense_mode_packet(packet):
    if packet is None or len(packet) != 84:
        raise ValueError(f"Unexpected packet length: {None if packet is None else len(packet)}")
    # Sync and checksum
    if ((packet[0] & 0xF0) >> 4 != 0xA) or ((packet[1] & 0xF0) >> 4 != 0x5):
        raise ValueError("Invalid sync bytes")
    chk_recv = ((packet[1] & 0x0F) << 4) | (packet[0] & 0x0F)
    chk_calc = 0
    for b in packet[2:]: chk_calc ^= b
    if chk_recv != chk_calc:
        raise ValueError("Checksum mismatch")

    start_q6 = packet[2] | ((packet[3] & 0x7F) << 8)
    start_angle = start_q6 / 64.0
    distances, angles = [], []
    for i in range(40):
        idx = 4 + 2*i
        d = (packet[idx] | (packet[idx+1] << 8)) / 1000.0
        a = (start_angle + i * 360/81/40.0 + ANGLE_CORRECTION) % 360
        distances.append(d)
        angles.append(a)
    return np.array(distances), np.array(angles)


def full_scan_new(sock):
    # Initialize high-res scan array
    full_angles = np.linspace(0, 360, LIDAR_LEN*2, endpoint=False)
    final_d = np.full_like(full_angles, np.inf, dtype=float)
    # Continue until <100 gaps in 0°–180°
    mask = (full_angles >= 0) & (full_angles <= 180)
    while np.sum(np.isinf(final_d[mask])) > 100:
        raw = receive_full_data(sock, 84)
        dists, angs = decode_dense_mode_packet(raw)
        # Sort by angle
        order = np.argsort(angs)
        angs, dists = angs[order], dists[order]
        # Map into final_d
        idxs = np.searchsorted(full_angles, angs)
        valid = idxs < len(final_d)
        final_d[idxs[valid]] = dists[valid]
    final_d[final_d==0] = np.inf
    # Interpolate
    x = np.arange(len(final_d))
    good = np.isfinite(final_d)
    final_d = np.interp(x, x[good], final_d[good]) + DISTANCE_CORRECTION
    # Return paired to user resolution
    return final_d[:LIDAR_LEN], full_angles[:LIDAR_LEN]


def save_scan_image(angles, distances, filename):
    # Polar to Cartesian
    rads = np.deg2rad(angles)
    x = distances * np.cos(rads)
    y = distances * np.sin(rads)
    plt.figure(figsize=(6,6))
    plt.scatter(x, y, s=1)
    plt.axis('equal')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('LIDAR Scan')
    plt.savefig(filename, dpi=300)
    plt.close()


def main():
    sock = connect_lidar()
    print('Initialization complete')
    sock, model, fw_min, fw_maj, hw, serial = get_info(sock)
    print(f"LIDAR Info - Model: {model}, FW: {fw_maj}.{fw_min}, HW: {hw}, SN: {serial}")
    status, err = get_health(sock)
    print(f"Health - Status: {status}, Error Code: {err}")
    start_scan(sock)

    fps_window = deque(maxlen=10)
    frame = 0
    try:
        while True:
            t0 = time.time()
            dists, angs = full_scan_new(sock)
            fname = f"scan_{frame:03d}.jpg"
            save_scan_image(angs, dists, fname)
            print(f"Saved {fname}")
            fps = 1.0 / (time.time() - t0)
            fps_window.append(fps)
            if len(fps_window)==fps_window.maxlen:
                print(f"Avg FPS: {sum(fps_window)/len(fps_window):.2f}")
            frame += 1
    except KeyboardInterrupt:
        print('Stopping...')
        stop_scan(sock)
        sock.close()

if __name__ == '__main__':
    main()
