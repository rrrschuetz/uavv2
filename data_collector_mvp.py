import socket
import struct
import pygame
import time
import numpy as np
import cv2
from picamera2 import Picamera2
from multiprocessing import Process
import threading
from collections import deque
from adafruit_pca9685 import PCA9685
from board import SCL, SDA
import busio
import matplotlib.pyplot as plt

# Servo functions
def set_servo_angle(pca, channel, angle):
    pulse_min = 260  # Pulse width for 0 degrees
    pulse_max = 380  # Pulse width for 180 degrees
    pulse_width = pulse_min + angle * (pulse_max - pulse_min)
    pca.channels[channel].duty_cycle = int(pulse_width / 4096 * 0xFFFF)


# ESC functions
def set_motor_speed(pca, channel, speed):
    pulse_min = 310  # Pulse width for 0% speed
    pulse_max = 409  # Pulse width for 100% speed
    pulse_width = pulse_min + speed * (pulse_max - pulse_min)
    pca.channels[channel].duty_cycle = int(pulse_width / 4096 * 0xFFFF)


def arm_esc(pca, channel):
    print("Arming ESC...")
    set_motor_speed(pca, channel, 0)
    time.sleep(1)
    print("ESC armed")


# LIDAR functions
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
    status, error_code = struct.unpack('<BH', response[3:6])
    return status, error_code


def get_info(sock):
    sock.send(b'\xA5\x50')
    response = receive_full_data(sock, 27)
    model, firmware_minor, firmware_major, hardware, serialnum = struct.unpack('<BBBB16s', response[7:])
    serialnum_str = serialnum[::-1].hex()
    return model, firmware_minor, firmware_major, hardware, serialnum_str


def start_scan(sock):
    sock.send(b'\xA5\x82\x05\x00\x00\x00\x00\x00\x22')
    response = receive_full_data(sock, 10)


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
        raise ValueError(
            f"Checksum validation failed: received={checksum_received:#04x}, computed={checksum_computed:#04x}")

    distances = []
    angles = []

    start_angle_q6 = (packet[2] | ((packet[3] & 0x7f) << 8))
    start_angle = start_angle_q6 / 64.0
    angle_diff = start_angle - old_start_angle

    if angle_diff > 0:
        valid_flag = True
        for i in range(40):
            index = 4 + i * 2
            if index + 1 >= len(packet):
                raise ValueError(f"Packet is too short for expected data: index {index}")
            distance = (packet[index] | (packet[index+1] << 8))
            distance /= 1000.0  # Convert from millimeters to meters
            angle = start_angle + i * 4.6 / 40.0

            distances.append(distance)
            angles.append(angle)
    else:
        valid_flag = False

    return {
        "valid": valid_flag,
        "start_angle": start_angle,
        "distances": distances,
        "angles": angles
    }


def full_scan(sock):
    old_start_angle = 0.0
    all_distances = []
    all_angles = []

    i = 0
    while True:
        data = receive_full_data(sock, 84)
        decoded_data = decode_dense_mode_packet(data, old_start_angle)
        if decoded_data['valid'] == False:
            break
        old_start_angle = decoded_data['start_angle']
        all_distances.extend(decoded_data['distances'])
        all_angles.extend(decoded_data['angles'])
        i += 1

    print(i,old_start_angle)
    return all_distances, all_angles


def lidar_thread(sock):
    fps_list = deque(maxlen=10)
    while True:
        start_time = time.time()
        distances, angles = full_scan(sock)
        end_time = time.time()

        data = np.column_stack((distances, angles))
        np.savetxt("radar.txt", data[1620:], header="Distances, Angles", comments='', fmt='%f')

        # Convert distances to a formatted string with 4 decimal places
        formatted_distances = ",".join(f"{d:.4f}" for d in distances[1620:])
        with open("lidar_distances.txt", "a") as file:
            file.write(formatted_distances + "\n")

        frame_time = end_time - start_time
        fps_list.append(1.0 / frame_time)

        moving_avg_fps = sum(fps_list) / len(fps_list)
        print(f'LIDAR moving average FPS: {moving_avg_fps:.2f}')


# Camera functions
def gamma_correction(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def enhance_lighting(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v = clahe.apply(v)
    hsv_enhanced = cv2.merge([h, s, v])
    enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
    return enhanced

def preprocess_image(image):
    gamma_corrected = gamma_correction(image)
    enhanced = enhance_lighting(gamma_corrected)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    return hsv

def apply_morphological_operations(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=4)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)
    return mask

def remove_small_contours(mask, min_area=2000):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            cv2.drawContours(mask, [contour], -1, 0, -1)
    return mask

def filter_contours(contours, min_area=2000, aspect_ratio_range=(1.5, 3.0), angle_range=(80, 100)):
    filtered_contours = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        width, height = rect[1]
        angle = rect[2]
        if width < height:
            width, height = height, width
            angle += 90
        aspect_ratio = width / height
        if aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1] and angle_range[0] <= angle <= angle_range[1]:
            filtered_contours.append(box)
    return filtered_contours

def detect_and_label_blobs(image):
    hsv = preprocess_image(image)

    # Adaptive color ranges for red and green detection
    red_lower1 = np.array([0, 50, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 50, 50])
    red_upper2 = np.array([180, 255, 255])
    green_lower1 = np.array([35, 40, 40])
    green_upper1 = np.array([70, 255, 255])
    green_lower2 = np.array([70, 40, 40])
    green_upper2 = np.array([90, 255, 255])

    # Create masks
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    green_mask1 = cv2.inRange(hsv, green_lower1, green_upper1)
    green_mask2 = cv2.inRange(hsv, green_lower2, green_upper2)
    green_mask = cv2.bitwise_or(green_mask1, green_mask2)

    # Apply morphological operations and remove small contours
    red_mask = remove_small_contours(apply_morphological_operations(red_mask))
    green_mask = remove_small_contours(apply_morphological_operations(green_mask))

    # Combine masks
    combined_mask = cv2.bitwise_or(red_mask, green_mask)

    # Find and filter contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = filter_contours(contours)

    red_x_coords = np.zeros(image.shape[1], dtype=float)
    green_x_coords = np.zeros(image.shape[1], dtype=float)

    for box in filtered_contours:
        rect = cv2.minAreaRect(box)
        center = (int(rect[0][0]), int(rect[0][1]))
        label = "R" if np.any(red_mask[center[1], center[0]]) else "G"
        left_end = min(box[:, 0])
        right_end = max(box[:, 0])
        if label == "R":
            red_x_coords[left_end:right_end] = 1.0
        else:
            green_x_coords[left_end:right_end] = 1.0

        # Draw and label the contours
        cv2.drawContours(image, [box], -1, (0, 255, 255), 2)
        cv2.putText(image, label, center, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 255), 2)

    return red_x_coords, green_x_coords, image


def camera_thread(picam0, picam1):
    fps_list = deque(maxlen=10)
    while True:
        start_time = time.time()
        image0 = picam0.capture_array()
        image1 = picam1.capture_array()
        image0_flipped = cv2.flip(image0, 0)
        image1_flipped = cv2.flip(image1, 0)
        combined_image = np.hstack((image1_flipped, image0_flipped))
        height = combined_image.shape[0]
        cropped_image = combined_image[height // 3:, :]
        red_x_coords, green_x_coords, image = detect_and_label_blobs(cropped_image)
        end_time = time.time()

        # Save the image with labeled contours
        #timestamp = time.strftime("%Y%m%d-%H%M%S")
        #filename = f"labeled_image_{timestamp}.jpg"
        filename = "labeled_image.jpg"
        cv2.imwrite(filename, image)

        frame_time = end_time - start_time
        fps_list.append(1.0 / frame_time)

        moving_avg_fps = sum(fps_list) / len(fps_list)
        #print(f'Camera moving average FPS: {moving_avg_fps:.2f}')


def xbox_controller_process(pca):
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("No joystick connected")
        pygame.quit()
        return

    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    print("Joystick initialized:")
    print(f"Name: {joystick.get_name()}")
    print(f"Number of axes: {joystick.get_numaxes()}")
    print(f"Number of buttons: {joystick.get_numbuttons()}")
    print(f"Number of hats: {joystick.get_numhats()}")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                #print(f"JOYAXISMOTION: axis={event.axis}, value={event.value}")
                if event.axis == 1:
                    set_motor_speed(pca, 13, abs(event.value * 0.3))
                elif event.axis == 2:
                    set_servo_angle(pca, 12, event.value * 0.4 + 0.5)
                if event.axis == 2:
                    set_servo_angle(pca, 2, event.value * 0.5 + 0.5)
            
            elif event.type == pygame.JOYBALLMOTION:
                print(f"JOYBALLMOTION: ball={event.ball}, rel={event.rel}")
            elif event.type == pygame.JOYBUTTONDOWN:
                print(f"JOYBUTTONDOWN: button={event.button}")
            elif event.type == pygame.JOYBUTTONUP:
                print(f"JOYBUTTONUP: button={event.button}")
            elif event.type == pygame.JOYHATMOTION:
                print(f"JOYHATMOTION: hat={event.hat}, value={event.value}")
            elif event.type == pygame.JOYDEVICEADDED:
                print(f"JOYDEVICEADDED: device_index={event.device_index}")
            elif event.type == pygame.JOYDEVICEREMOVED:
                print(f"JOYDEVICEREMOVED: instance_id={event.instance_id}")
            elif event.type == pygame.QUIT:
                print("QUIT event")
                return

        time.sleep(1 / 30)


def main():
    # Initialize the I2C bus
    i2c = busio.I2C(SCL, SDA)

    # Create a simple PCA9685 class instance
    pca = PCA9685(i2c)
    pca.frequency = 50  # Standard servo frequency
    arm_esc(pca,1)

    # LIDAR setup
    IP_ADDRESS = '192.168.11.2'
    PORT = 8089
    sock = connect_lidar(IP_ADDRESS, PORT)

    print('Getting LIDAR info...')
    info = get_info(sock)
    print('LIDAR Info:', info)

    print('Getting LIDAR health...')
    health = get_health(sock)
    print('LIDAR Health:', health)

    print('Starting scan...')
    start_scan(sock)

    # Camera setup
    picam0 = Picamera2(camera_num=0)
    picam1 = Picamera2(camera_num=1)
    config = {"format": 'RGB888', "size": (640, 400)}
    picam0.configure(picam0.create_preview_configuration(main=config))
    picam1.configure(picam1.create_preview_configuration(main=config))
    picam0.start()
    picam1.start()

    # Start processes
    lidar_thread_instance = threading.Thread(target=lidar_thread, args=(sock,))
    camera_thread_instance = threading.Thread(target=camera_thread, args=(picam0, picam1))
    xbox_controller_process_instance = Process(target=xbox_controller_process,args=(pca,))

    lidar_thread_instance.start()
    camera_thread_instance.start()
    xbox_controller_process_instance.start()

    try:
        #pass
        lidar_thread_instance.join()
        camera_thread_instance.join()
        xbox_controller_process_instance.join()

    except KeyboardInterrupt:
        picam0.stop()
        picam1.stop()
        stop_scan(sock)
        sock.close()
        pygame.quit()

if __name__ == '__main__':
    main()
