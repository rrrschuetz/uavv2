import socket
import struct
import serial
import pygame
import json
import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import trim_mean
import cv2
from picamera2 import Picamera2
from multiprocessing import Process, Value
import threading
from collections import deque
from adafruit_pca9685 import PCA9685
import board
from board import SCL, SDA
import busio
from gpiozero import Button
import qmc5883l as qmc5883
import time, sys
import usb.core, usb.util
import math, statistics
import torch
from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas
from torch.distributions.constraints import positive

from lidar_color_model import CNNModel  # Import the model from model.py
# from preprocessing import preprocess_input, load_scaler  # Import preprocessing functions
from preprocessing_no_scaler import preprocess_input  # Import preprocessing functions

#########################################
WRITE_CAMERA_IMAGE = False
WRITE_CAMERA_MOVIE = False
TOTAL_LAPS = 1
PARKING_MODE = True
#########################################

# Configuration for WT61 Gyroscope
SERIAL_PORT = "/dev/ttyAMA0"  # or "/dev/ttyS0" if you have mapped accordingly
BAUD_RATE = 115200
TIMEOUT = 0.5  # Set a slightly longer timeout to ensure full packet reads

SENSOR_PIN = 17

LIDAR_LEN = 1620
COLOR_LEN = 1280
ANGLE_CORRECTION = 180.0
DISTANCE_CORRECTION = -0.10

SERVO_FACTOR = 0.5  # 0.4
SERVO_BASIS = 0.5  # 1.55 # 0.55
MOTOR_FACTOR = 0.45  # 0.3
MOTOR_BASIS = 0.1

RACE_SPEED = -0.35
EMERGENCY_SPEED = -0.45
PARK_SPEED = -0.35  # -0.55
PARK_STEER = 2.5
PARK_ANGLE = 90

# Global variables
Gclock_wise = False
Glidar_string = ""
Gcolor_string = ",".join(["0"] * COLOR_LEN)
Gx_coords = np.zeros(COLOR_LEN, dtype=float)
Gline_orientation = None
Gobstacles = True
Gpitch = 0.0
Groll = 0.0
Gyaw = 0.0
Gaccel_x = 0.0
Gaccel_y = 0.0
Gaccel_z = 0.0
Gheading_estimate = 0.0
Gheading_start = 0.0
Glap_end = False
Glidar_moving_avg_fps = 0.0
Gcamera_moving_avg_fps = 0.0
shared_race_mode = Value('i', 0)

i2c = board.I2C()
qmc = qmc5883.QMC5883L(i2c)
qmc.output_data_rate = (qmc5883.OUTPUT_DATA_RATE_200)


# Servo functions
def set_servo_angle(pca, channel, angle):
    pulse_min = 260  # Pulse width for 0 degrees
    pulse_max = 380  # Pulse width for 180 degrees
    pulse_width = pulse_min + angle * (pulse_max - pulse_min)
    try:
        pca.channels[channel].duty_cycle = int(pulse_width / 4096 * 0xFFFF)
    except ValueError:
        print("Invalid angle value: ", angle)


# ESC functions
def set_motor_speed(pca, channel, speed):
    pulse_min = 310  # Pulse width for 0% speed
    pulse_max = 409  # Pulse width for 100% speed
    pulse_width = pulse_min + speed * (pulse_max - pulse_min)
    try:
        pca.channels[channel].duty_cycle = int(pulse_width / 4096 * 0xFFFF)
    except ValueError:
        print("Invalid speed value: ", speed)


def arm_esc(pca, channel):
    print("Arming ESC...")
    set_motor_speed(pca, channel, 0)
    time.sleep(1)
    print("ESC armed")


# LIDAR functions
IP_ADDRESS = '192.168.11.2'
PORT = 8089


def connect_lidar(ip=IP_ADDRESS, port=PORT):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect((ip, port))
    return sock


def receive_full_data(sock, expected_length, timeout=5):
    sock.settimeout(timeout)
    data = b''
    try:
        while len(data) < expected_length:
            try:
                packet, _ = sock.recvfrom(expected_length - len(data))
                data += packet
            except socket.timeout:
                print(f"Timeout after {timeout} seconds while waiting for data.")
                raise  # Re-raise the timeout exception
    except socket.timeout:
        print("Socket timed out. Reconnect.")
        return None  # Return None
    return data


def get_health(sock):
    sock.send(b'\xA5\x52')
    response = receive_full_data(sock, 10)
    status, error_code = struct.unpack('<BH', response[3:6])
    return status, error_code


def get_info(sock):
    sock.send(b'\xA5\x50')
    while True:
        response = receive_full_data(sock, 27)
        if response is not None: break
        sock.close()
        sock = connect_lidar()
        start_scan(sock)
    model, firmware_minor, firmware_major, hardware, serialnum = struct.unpack('<BBBB16s', response[7:])
    serialnum_str = serialnum[::-1].hex()
    return sock, model, firmware_minor, firmware_major, hardware, serialnum_str


def start_scan(sock):
    sock.send(b'\xA5\x82\x05\x00\x00\x00\x00\x00\x22')
    response = receive_full_data(sock, 10)


def stop_scan(sock):
    sock.send(b'\xA5\x25')
    time.sleep(0.1)


def decode_dense_mode_packet(packet):
    try:
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

        for i in range(40):
            index = 4 + i * 2
            if index + 1 >= len(packet):
                raise ValueError(f"Packet is too short for expected data: index {index}")
            distance = (packet[index] | (packet[index + 1] << 8))
            distance /= 1000.0  # Convert from millimeters to meters
            angle = start_angle + i * 360 / 81 / 40.0

            distances.append(distance)
            angles.append(angle)

        return {
            "start_angle": start_angle,
            "distances": distances,
            "angles": angles
        }

    except Exception as e:
        print(f"Error decoding lidar data: {e}")
        return None


def full_scan(sock):
    inf_threshold = 100  # Stop scanning when fewer than 100 np.inf values remain
    final_distances = np.full(LIDAR_LEN * 2, np.inf)  # Initialize with np.inf for missing values
    full_angle_range = np.linspace(0, 360, LIDAR_LEN * 2, endpoint=False)  # High-resolution angle range
    angle_0_to_180_indices = (full_angle_range >= 0) & (full_angle_range <= 180)

    # Continue scanning until the number of np.inf values is below the threshold
    while np.sum(np.isinf(final_distances[angle_0_to_180_indices])) > inf_threshold:

        # Collect data in each iteration (single scan)
        data = receive_full_data(sock, 84)
        decoded_data = decode_dense_mode_packet(data)
        if decoded_data == None: continue

        distances = np.array(decoded_data['distances'])
        angles = np.array(decoded_data['angles'])

        # Ensure all angles are within the 0-360 degree range and apply angle correction if necessary
        angles = (angles + ANGLE_CORRECTION) % 360

        # Sort all angles and distances based on the sorted angles
        sorted_indices = np.argsort(angles)
        sorted_angles = angles[sorted_indices]
        sorted_distances = distances[sorted_indices]

        # Update final_distances with the latest distance for each unique angle in full resolution
        for angle, distance in zip(sorted_angles, sorted_distances):
            # Find the closest index in the full resolution angle array
            index = np.searchsorted(full_angle_range, angle, side='left')
            if index < len(final_distances):
                final_distances[index] = distance  # Update the distance at the correct index

    # Replace zero distance values with np.inf (if zero means missing data)
    final_distances[final_distances == 0] = np.inf

    # Interpolate missing or infinite distances
    finite_vals = np.isfinite(final_distances)
    x = np.arange(len(final_distances))
    interpolated_distances = np.interp(x, x[finite_vals], final_distances[finite_vals])

    # Apply distance correction if necessary
    interpolated_distances += DISTANCE_CORRECTION

    # Save the processed data (angles and distances) to a file
    data = np.column_stack((interpolated_distances, full_angle_range))
    data = data[:LIDAR_LEN]
    # np.savetxt("radar.txt", data, header="Distances, Angles", comments='', fmt='%f')

    return interpolated_distances, full_angle_range


def navigate(sock, narrow=True):
    window_size = 10  # Adjust based on desired robustness
    input_size = window_size if narrow else 500
    min_distance = 3.0
    min_angle = 0.0
    left_min_distance = 3.0
    left_min_angle = 0.0
    right_min_distance = 3.0
    right_min_angle = 0.0

    interpolated_distances, angles = full_scan(sock)
    # Smooth the data using a median filter to reduce noise and outliers
    valid_distances = median_filter(interpolated_distances[:LIDAR_LEN], size=window_size)
    front_distance = np.max(valid_distances[LIDAR_LEN // 2 - input_size // 2:LIDAR_LEN // 2 + input_size // 2])
    # Calculate lidar orientation (clockwise ?)
    first_half_sum = np.sum(valid_distances[:LIDAR_LEN // 2])
    second_half_sum = np.sum(valid_distances[LIDAR_LEN // 2:])
    distance_ratio = first_half_sum / second_half_sum if second_half_sum != 0 else float('inf')

    # Use the sliding window to compute the local robust minimum distance
    for i in range(0, LIDAR_LEN - window_size + 1):
        window = valid_distances[i:i + window_size]
        trimmed_mean_distance = trim_mean(window, proportiontocut=0.1)
        if 0 < trimmed_mean_distance < min_distance:
            min_distance = trimmed_mean_distance
            min_index = i + window_size // 2  # Center of the window
            min_angle = angles[min_index]
        if i < LIDAR_LEN // 2:
            if 0 < trimmed_mean_distance < right_min_distance:
                right_min_distance = trimmed_mean_distance
                right_min_angle = angles[i + window_size // 2]
        else:
            if 0 < trimmed_mean_distance < left_min_distance:
                left_min_distance = trimmed_mean_distance
                left_min_angle = angles[i + window_size // 2]

    return {
        "min_distance": min_distance,
        "min_angle": min_angle,
        "left_min_distance": left_min_distance,
        "left_min_angle": left_min_angle,
        "right_min_distance": right_min_distance,
        "right_min_angle": right_min_angle,
        "front_distance": front_distance,
        "distance_ratio": distance_ratio
    }


def lidar_thread(sock, pca, shared_GX, shared_GY, shared_race_mode):
    global Glidar_string, Gcolor_string
    global Gx_coords
    global Glidar_moving_avg_fps
    global Gobstacles

    model_cc = None
    model_cw = None
    # scaler_lidar = None
    device = None

    fps_list = deque(maxlen=10)
    while True:
        start_time = time.time()

        if shared_race_mode.value == 4:  # Training
            # print("LIDAR in manual mode")
            interpolated_distances, angles = full_scan(sock)
            Glidar_string = ",".join(f"{d:.4f}" for d in interpolated_distances[:LIDAR_LEN])
            with open("data_file.txt", "a") as file:
                file.write(f"{shared_GX.value},{shared_GY.value},{Glidar_string},{Gcolor_string}\n")

        elif shared_race_mode.value == 1:  # Race
            # print("LIDAR in autonomous mode")
            interpolated_distances, angles = full_scan(sock)

            if model_cc is None:
                # Load the trained model and the scaler
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # Initialize the model
                model_cc = CNNModel(LIDAR_LEN, COLOR_LEN).to(device)
                model_cw = CNNModel(LIDAR_LEN, COLOR_LEN).to(device)

                # Load the trained weights into the model
                if Gobstacles:
                    state_dict_cc = torch.load('./model_cc.pth', map_location=torch.device('cpu'))
                    state_dict_cw = torch.load('./model_cw.pth', map_location=torch.device('cpu'))
                else:
                    state_dict_cc = torch.load('./model_cc_o.pth', map_location=torch.device('cpu'))
                    state_dict_cw = torch.load('./model_cw_o.pth', map_location=torch.device('cpu'))

                # Convert all weights to float32 if they are in float64
                for key, value in state_dict_cc.items():
                    if value.dtype == torch.float64:  # Check if the parameter is in double precision
                        state_dict_cc[key] = value.float()  # Convert to single precision (float32)
                for key, value in state_dict_cw.items():
                    if value.dtype == torch.float64:  # Check if the parameter is in double precision
                        state_dict_cw[key] = value.float()  # Convert to single precision (float32)

                # Load the state dict into the model
                model_cc.load_state_dict(state_dict_cc)
                model_cw.load_state_dict(state_dict_cw)
                model_cc.eval()
                model_cw.eval()

            # if scaler_lidar is None:
            #    # Load the scaler for LIDAR data
            #    scaler_lidar = load_scaler('./scaler.pkl')

            # emergency break
            left_min = min(interpolated_distances[LIDAR_LEN // 8 * 3: LIDAR_LEN // 2])
            right_min = min(interpolated_distances[LIDAR_LEN // 2: LIDAR_LEN // 8 * 5])
            if 0 < left_min < 0.07 or 0 < right_min < 0.07:
                print("Emergency break")
                dir = 1 if left_min < right_min else -1
                set_servo_angle(pca, 12, SERVO_BASIS + PARK_STEER * SERVO_FACTOR * dir)
                time.sleep(0.5)
                set_motor_speed(pca, 13, MOTOR_BASIS - EMERGENCY_SPEED * MOTOR_FACTOR)
                time.sleep(0.5)
                set_motor_speed(pca, 13, MOTOR_BASIS)
                set_servo_angle(pca, 12, SERVO_BASIS)
                time.sleep(0.5)
                continue

            ld = interpolated_distances[:LIDAR_LEN]
            # if Gclock_wise:
            #    ld = ld[::-1]
            # lidar_tensor, color_tensor = preprocess_input(
            #    ld, Gx_coords, scaler_lidar, device)
            lidar_tensor, color_tensor = preprocess_input(
                ld, Gx_coords, device)

            if lidar_tensor is not None and color_tensor is not None:
                # Perform inference
                with torch.no_grad():
                    if not Gclock_wise:
                        output = model_cc(lidar_tensor, color_tensor)
                    else:
                        output = model_cw(lidar_tensor, color_tensor)

                # Convert the model's output to steering commands or other UAV controls
                steering_commands = output.cpu().numpy()
                # print("Steering Commands:", steering_commands)
                X = steering_commands[0, 0]  # Extract GX (first element of the output)
                # Y = steering_commands[0, 1]  # Extract GY (second element of the output)
                Y = RACE_SPEED
                # if Gclock_wise:
                #    X = -X
                if -1.0 < X < 1.0 and -1.0 < Y < 0.0:
                    set_servo_angle(pca, 12, X * SERVO_FACTOR + SERVO_BASIS)
                    set_motor_speed(pca, 13, Y * MOTOR_FACTOR + MOTOR_BASIS)
                else:
                    pass
                    print("Invalid steering commands:", X, Y)

        elif shared_race_mode.value in [0, 2]:
            # print("Lidar inactve")
            time.sleep(1)

        frame_time = time.time() - start_time
        fps_list.append(1.0 / frame_time)

        Glidar_moving_avg_fps = sum(fps_list) / len(fps_list)
        # print(f'LIDAR moving average FPS: {Glidar_moving_avg_fps:.2f}')


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


def remove_small_contours(mask, min_area=500):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            cv2.drawContours(mask, [contour], -1, 0, -1)
    return mask


def filter_contours(contours, min_area=500, aspect_ratio_range=(1.0, 4.0), angle_range=(80, 100)):
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


def check_line_thickness(line, mask, min_thickness):
    """Check if the line has the specified minimum thickness."""
    x1, y1, x2, y2 = line
    length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Calculate unit vector perpendicular to the line
    dx = (y2 - y1) / length
    dy = -(x2 - x1) / length

    # Count consecutive non-zero pixels along the perpendicular direction
    thickness_count = 0
    for offset in range(-min_thickness, min_thickness + 1):
        cx = int((x1 + x2) / 2 + dx * offset)
        cy = int((y1 + y2) / 2 + dy * offset)

        # Ensure point is within bounds
        if 0 <= cx < mask.shape[1] and 0 <= cy < mask.shape[0]:
            if mask[cy, cx] > 0:
                thickness_count += 1

    # print(f"Line thickness: {thickness_count}")
    return thickness_count >= min_thickness


def detect_and_label_blobs(image, num_detector_calls):
    global Gclock_wise

    first_line = False
    second_line = False
    line_orientation = ""
    magenta_rectangle = False

    hsv = preprocess_image(image)
    # cv2.imwrite('hsv.jpg', hsv)

    # Adaptive color ranges for red and green detection
    red_lower1 = np.array([0, 50, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 50, 50])
    red_upper2 = np.array([180, 255, 255])

    green_lower1 = np.array([35, 40, 40])
    green_upper1 = np.array([70, 255, 255])
    green_lower2 = np.array([70, 40, 40])
    green_upper2 = np.array([90, 255, 255])

    blue_lower = np.array([90, 70, 90])  # HSV range for blue detection
    blue_upper = np.array([140, 255, 255])

    amber_lower = np.array([10, 50, 50])  # Lower bound for hue, saturation, and brightness
    amber_upper = np.array([20, 255, 255])  # Upper bound for hue, saturation, and brightness

    magenta_lower = np.array([140, 50, 50])  # HSV range for magenta color detection
    magenta_upper = np.array([170, 255, 255])

    # Detect red and green blocks
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

    x_coords = np.zeros(image.shape[1], dtype=float)

    for box in filtered_contours:
        rect = cv2.minAreaRect(box)
        center = (int(rect[0][0]), int(rect[0][1]))
        label = "R" if np.any(red_mask[center[1], center[0]]) else "G"
        left_end = min(box[:, 0])
        right_end = max(box[:, 0])
        if label == "R":
            x_coords[left_end:right_end] = 1.0
        else:
            x_coords[left_end:right_end] = -1.0

        # Draw and label the contours
        cv2.drawContours(image, [box], -1, (0, 255, 255), 2)
        cv2.putText(image, label, center, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 2)

    if (num_detector_calls % 2 == 0):

        # Detect blue and amber lines
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        blue_mask = remove_small_contours(blue_mask)
        amber_mask = cv2.inRange(hsv, amber_lower, amber_upper)
        amber_mask = remove_small_contours(amber_mask)

        if Gclock_wise:
            first_line_mask = amber_mask
            second_line_mask = blue_mask
        else:
            first_line_mask = blue_mask
            second_line_mask = amber_mask

        most_significant_line = None
        max_line_length = 0
        height, width = image.shape[:2]

        lines = cv2.HoughLinesP(first_line_mask, 1, np.pi / 180, threshold=250, minLineLength=150, maxLineGap=1)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if (y1 > height // 2 and y2 > height // 2
                        and check_line_thickness(line[0], first_line_mask, 6)):
                    len = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if len > max_line_length:
                        max_line_length = len
                        most_significant_line = line[0]

        if most_significant_line is not None:
            first_line = True
            x1, y1, x2, y2 = most_significant_line
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 5)

            # Determine the orientation of the line based on endpoint positions
            if x1 < width // 4 * 1: y1 += (height // 10)  # Camera view angle correction left sight
            if x2 > width // 4 * 3: y2 += (height // 10)  # Camera view angle correction right sight
            if x1 < x2:
                line_orientation = "UP" if y1 > y2 else "DOWN"
            else:
                line_orientation = "UP" if y2 < y1 else "DOWN"
            # print(f"First line endpoints: ({x1}, {y1}), ({x2}, {y2})")
            # print(f"First line orientation: {line_orientation}")

        lines = cv2.HoughLinesP(second_line_mask, 1, np.pi / 180, threshold=250, minLineLength=150, maxLineGap=1)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if (y1 > height // 2 and y2 > height // 2
                        and check_line_thickness(line[0], second_line_mask, 6)):
                    second_line = True
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 5)

        # Detect magenta parking lot
        magenta_mask = cv2.inRange(hsv, magenta_lower, magenta_upper)
        magenta_mask = remove_small_contours(apply_morphological_operations(magenta_mask))

        # Find and filter contours for magenta blobs only with outwards looking camera
        contours, _ = cv2.findContours(magenta_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # 5000 size of parking lot
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                left_end = min(box[:, 0])
                right_end = max(box[:, 0])
                if (not Gclock_wise and left_end > COLOR_LEN / 2) or (Gclock_wise and right_end < COLOR_LEN / 2):
                    # print(f"Magenta rectangle detected: {area} pixels")
                    magenta_rectangle = True
                    cv2.drawContours(image, [contour], -1, (255, 255, 255), 2)  # Draw the magenta rectangle

        # Add timestamp in the lower left corner
        timestamp = time.strftime("%H:%M:%S", time.localtime()) + f":{int((time.time() % 1) * 100):02d}"
        cv2.putText(image, timestamp, (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if WRITE_CAMERA_IMAGE and num_detector_calls % 2 == 0:
        cv2.imwrite("labeled_image.jpg", image)
        cv2.imwrite('amber_mask.jpg', amber_mask)
        cv2.imwrite('blue_mask.jpg', blue_mask)
        # cv2.imwrite('magenta_mask.jpg', magenta_mask)
        # cv2.imwrite('red_mask.jpg', red_mask)
        # cv2.imwrite('green_mask.jpg', green_mask)

    return x_coords, first_line, second_line, magenta_rectangle, line_orientation, image


def camera_thread(pca, picam0, picam1, shared_race_mode, device):
    global Gcolor_string, Gx_coords
    global Gline_orientation
    global Glap_end, Gheading_estimate  # heading
    global Gcamera_moving_avg_fps, Glidar_moving_avg_fps

    fps_list = deque(maxlen=10)
    frame_height, frame_width, _ = picam0.capture_array().shape
    frame_width *= 2
    frame_height //= 2
    print(f"Frame width: {frame_width}, Frame height: {frame_height}")

    # VideoWriter setup
    if WRITE_CAMERA_MOVIE:
        fps = 20  # Set frames per second for the output video
        video_filename = "output_video_000.avi"  # Output video file name
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for the output video file
        video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))
        frame_count = 0
        file_index = 0
        max_frame_count = 2000  # Maximum number of frames per video file

    num_detector_calls = 0
    num_laps = 0
    parking_lot_reached = False

    try:
        while True:
            num_detector_calls += 1
            if shared_race_mode.value in [0, 1, 3, 4]:
                start_time = time.time()
                image0 = picam0.capture_array()
                image1 = picam1.capture_array()
                image0_flipped = cv2.flip(image0, -1)
                image1_flipped = cv2.flip(image1, -1)
                combined_image = np.hstack((image0_flipped, image1_flipped))
                cropped_image = combined_image[frame_height:, :]
                Gx_coords, first_line, second_line, parking_lot, line_orientation, image \
                    = detect_and_label_blobs(cropped_image, num_detector_calls)

                # if Gclock_wise:
                #    Gx_coords = Gx_coords * -1.0
                #    Gx_coords = Gx_coords[::-1]
                Gcolor_string = ",".join(map(str, Gx_coords.astype(int)))

                if first_line and Gline_orientation is None: Gline_orientation = line_orientation

                if shared_race_mode.value == 1:
                    # blank_led(device)

                    if PARKING_MODE:
                        parking_lot_reached = parking_lot_reached or parking_lot
                        # if parking_lot_reached: print("Parking lot reached.")

                        if parking_lot_reached:
                            parking_led(device)
                        else:
                            blank_led(device)

                        if Glap_end and parking_lot_reached:
                            parking_lot_reached = False
                            num_laps += 1
                            blank_led(device)
                        else:
                            if parking_lot_reached and num_laps >= TOTAL_LAPS:
                                shared_race_mode.value = 2
                                print(f"Laps completed: {num_laps} / {Gheading_estimate:.2f}")
                                print(f'LIDAR moving average FPS: {Glidar_moving_avg_fps:.2f}')
                                print(f'Camera moving average FPS: {Gcamera_moving_avg_fps:.2f}')
                                print("Parking initiated")
                    else:
                        if Glap_end: num_laps += 1  # here is something missing !
                        if num_laps >= TOTAL_LAPS:
                            shared_race_mode.value = 2
                            print("Race completed.")

                # Save the image with labeled contours
                if WRITE_CAMERA_MOVIE:
                    video_writer.write(image)
                    frame_count += 1

                    if frame_count > max_frame_count:
                        video_writer.release()
                        file_index += 1
                        frame_count = 0
                        video_filename = f"output_video_{file_index:03d}.avi"
                        video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

                # time.sleep(0.05)
                frame_time = time.time() - start_time
                fps_list.append(1.0 / frame_time)

                Gcamera_moving_avg_fps = sum(fps_list) / len(fps_list)
                # print(f'Camera moving average FPS: {Gcamera_moving_avg_fps:.2f}')

            else:
                # print("Camera inactive")
                time.sleep(1.0)

    except KeyboardInterrupt:
        print("Keyboard Interrupt detected, stopping video capture and saving...")

    finally:
        if WRITE_CAMERA_MOVIE and video_writer.isOpened():
            video_writer.release()


def xbox_controller_process(pca, shared_GX, shared_GY, shared_race_mode):
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
                # print(f"JOYAXISMOTION: axis={event.axis}, value={event.value}")
                if event.axis == 1:
                    shared_GY.value = event.value
                    set_motor_speed(pca, 13, event.value * MOTOR_FACTOR + MOTOR_BASIS)
                elif event.axis == 2:
                    shared_GX.value = event.value
                    set_servo_angle(pca, 12, event.value * SERVO_FACTOR + SERVO_BASIS)
                elif event.axis == 3:
                    pass
                    # set_servo_angle(pca, 11, abs(event.value) * 1.5 + 0.0)

            elif event.type == pygame.JOYBALLMOTION:
                print(f"JOYBALLMOTION: ball={event.ball}, rel={event.rel}")

            elif event.type == pygame.JOYBUTTONDOWN:
                print(f"JOYBUTTONDOWN: button={event.button}")
                if event.button == 0:  # A button
                    if shared_race_mode.value == 0:
                        print("Race started")
                        shared_race_mode.value = 3
                elif event.button == 1:  # B button
                    if shared_race_mode.value == 0:
                        print("Training started")
                        shared_race_mode.value = 4
                elif event.button == 3:  # X button
                    print("STOP")
                    shared_race_mode.value = 0
                    set_motor_speed(pca, 13, MOTOR_BASIS)
                    set_servo_angle(pca, 12, SERVO_BASIS)
                elif event.button == 4:  # Y button
                    print("Y Button pressed")

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


def yaw_difference(yaw1, yaw2):
    """Calculate the shortest difference between two yaw angles in degrees."""
    diff = yaw2 - yaw1
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff


# Get magnetometer heading
def vector_2_degrees(x, y):
    angle = math.degrees(math.atan2(y, x))
    if angle < 0:
        angle += 360
    return angle

# Tilt compensation for magnetometer using pitch and roll
def tilt_compensate(mag_x, mag_y, mag_z, pitch, roll):
    mag_x_comp = mag_x * math.cos(pitch) + mag_z * math.sin(pitch)
    mag_y_comp = (mag_x * math.sin(roll) * math.sin(pitch)
                  + mag_y * math.cos(roll)
                  - mag_z * math.sin(roll) * math.cos(pitch))
    return mag_x_comp, mag_y_comp

# Cache
Gcalibration_cache = None
def load_compass_calibration(filename="compass_calibration.json"):
    global Gcalibration_cache
    if Gcalibration_cache is None:
        try:
            with open(filename, "r") as f:
                data = json.load(f)
            offsets = data.get("offsets", [0,0,0])
            scales  = data.get("scales",  [1,1,1])
            Gcalibration_cache = (offsets, scales)
            print("Calibration data loaded.")
        except Exception as e:
            print(f"Error loading calibration data: {e}")
            Gcalibration_cache = ([0,0,0], [1,1,1])
    return Gcalibration_cache

def calibrate_magnetometer_data(raw_data, offsets, scales):
    """
    Wendet Hard-Iron-Offset und Soft-Iron-Skalierung an:
      calibrated = (raw - offset) * scale
    """
    calibrated = []
    for raw, offset, scale in zip(raw_data, offsets, scales):
        calibrated.append((raw - offset) * scale)
    return tuple(calibrated)

def tilt_compensate(x, y, z, pitch_rad, roll_rad):
    """
    Tilt-Compensation gemäß:
      Xh = x*cos(pitch) + z*sin(pitch)
      Yh = x*sin(roll)*sin(pitch) + y*cos(roll) - z*sin(roll)*cos(pitch)
    """
    Xh = x*math.cos(pitch_rad) + z*math.sin(pitch_rad)
    Yh = x*math.sin(roll_rad)*math.sin(pitch_rad) + \
         y*math.cos(roll_rad) - \
         z*math.sin(roll_rad)*math.cos(pitch_rad)
    return Xh, Yh

def vector_2_degrees(x, y):
    """Wandelt (X,Y) in einen 0–360° Heading um."""
    heading = math.atan2(y, x)
    heading_deg = (math.degrees(heading) + 360) % 360
    return heading_deg

def get_magnetometer_heading():
    offsets, scales = load_compass_calibration()
    retries = 10
    for _ in range(retries):
        try:
            raw = qmc.magnetic  # (x_raw, y_raw, z_raw)
            # 1) Offset + Scale (hier multiplizieren!)
            mag_x, mag_y, mag_z = calibrate_magnetometer_data(raw, offsets, scales)
            # 2) Tilt-Compensation
            pitch_rad = math.radians(Gpitch)
            roll_rad  = math.radians(Groll)
            mag_xc, mag_yc = tilt_compensate(mag_x, mag_y, mag_z, pitch_rad, roll_rad)
            # 3) Heading berechnen
            return vector_2_degrees(mag_xc, mag_yc)
        except OSError:
            time.sleep(0.5)
    # Fallback
    return None


def parse_wt61_data(data):
    if len(data) == 11 and data[0] == 0x55 and sum(data[0:-1]) & 0xFF == data[-1]:
        data_type = data[1]
        values = struct.unpack('<hhh', data[2:8])  # Convert data to three signed short values
        return data_type, values
    else:
        # print("Invalid data package")
        return None, (None, None, None)


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
            response = ser.read(20)
            if response:
                print(f"Received response: {response.hex()}")

    except serial.SerialException as e:
        print(f"Serial error: {e}")


def orientation(angle):
    mod = angle % 90
    if 0 <= mod < 45:
        return mod
    else:
        return mod - 90


def gyro_thread(shared_race_mode):
    global Gaccel_x, Gaccel_y, Gaccel_z
    global Gpitch, Groll, Gyaw
    global Gheading_estimate, Gheading_start, Glap_end

    buff = bytearray()  # Buffer to store incoming serial data
    packet_counter = 0  # Counter to skip packets

    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT) as ser:
            ser.reset_input_buffer()  # Clear old data at the start

            while True:
                if ser.in_waiting:
                    # Read all available data and append it to the buffer
                    data = ser.read(ser.in_waiting)
                    buff.extend(data)  # Add data to the buffer

                    # Process all available packets in the buffer
                    while len(buff) >= 11:  # While we have at least one full packet

                        if buff[0] == 0x55 and buff[1] in [0x51, 0x53]:  # Valid start byte for packet
                            packet = buff[:11]  # Get a full 11-byte packet
                            buff = buff[11:]  # Remove the processed packet from the buffer

                            # Increment packet counter and skip processing unless it's every 5th packet
                            packet_counter += 1
                            if packet_counter % 5 != 0:
                                continue  # Skip this packet

                            # Parse the packet
                            data_type, values = parse_wt61_data(packet)

                            # Handle accelerometer data (0x51)
                            if data_type == 0x51:
                                accel = [v / 32768.0 * 16 for v in values]  # Convert to G
                                Gaccel_x, Gaccel_y, Gaccel_z = accel

                            # Handle gyroscope data (0x53)
                            elif data_type == 0x53:
                                gyro = [v / 32768.0 * 180 for v in values]  # Convert to degrees
                                Gpitch, Groll, Gyaw = gyro
                                # print(f"Gpitch, Groll, Gyaw: {Gpitch} {Groll} {Gyaw}")

                        else:
                            buff.pop(0)  # Remove one byte and continue checking

                else:
                    # Get the magnetometer heading (absolute heading)
                    Gheading_estimate = get_magnetometer_heading()
                    yaw_diff = abs(yaw_difference(Gheading_estimate, Gheading_start))
                    print(f"... Gheading_estimate: {Gheading_estimate:.2f}")
                    Glap_end = yaw_diff < 10
                    time.sleep(0.1)

    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except KeyboardInterrupt:
        print("Stopping data read.")


def align_parallel(pca, sock, shared_race_mode, stop_distance=1.4, min_yaw=10):
    global Gheading_estimate, Gheading_start

    distance2stop = 1.0
    yaw_diff = 90
    while shared_race_mode.value == 2 and (abs(yaw_diff) > min_yaw or distance2stop > 0):
        yaw_diff = orientation(yaw_difference(Gheading_start, Gheading_estimate))
        steer = abs(yaw_diff) / 90 * 1  # 1.7
        if yaw_diff > 0: steer = - steer
        steer = max(min(steer, 1), -1)
        narrow = abs(yaw_diff) < min_yaw
        position = navigate(sock, narrow)
        front_distance = position['front_distance']
        distance2stop = front_distance - stop_distance
        print(f"Gheading_start: {Gheading_start} Gheading_estimate: {Gheading_estimate} yaw_diff: {yaw_diff}")
        print(
            f"{narrow} front_distance: {front_distance:.2f} stop_distance: {stop_distance:.2f} distance2stop: {distance2stop:.2f} steer: {steer}")
        set_servo_angle(pca, 12, PARK_STEER * steer * SERVO_FACTOR + SERVO_BASIS)
        set_motor_speed(pca, 13, PARK_SPEED * MOTOR_FACTOR + MOTOR_BASIS)
        time.sleep(0.05)

    set_motor_speed(pca, 13, MOTOR_BASIS)
    set_servo_angle(pca, 12, SERVO_BASIS)
    print(f"Car aligned")


def align_angular(pca, sock, angle, shared_race_mode):
    global Gheading_estimate

    yaw_init = Gheading_estimate
    print(f"Car alignment: initial angle {yaw_init:.2f} delta angle {angle:.2f}")
    while shared_race_mode.value == 2 and abs(yaw_difference(Gheading_estimate, yaw_init)) < abs(angle):
        # print(f"Car orthogonal alignment: angle {yaw_difference(Gheading_estimate, yaw_init):.2f}")
        steer = 1 - abs(yaw_difference(Gheading_estimate, yaw_init)) / abs(angle)
        if angle > 0: steer = -steer
        steer = max(min(steer, 1), -1)
        set_servo_angle(pca, 12, PARK_STEER * steer * SERVO_FACTOR + SERVO_BASIS)
        set_motor_speed(pca, 13, PARK_SPEED * MOTOR_FACTOR + MOTOR_BASIS)
        time.sleep(0.05)
        position = navigate(sock)
        # print(f"Front distance: {position['front_distance']:.2f}")
        if position['front_distance'] < 0.05: break

    # set_servo_angle(pca, 13, MOTOR_BASIS)
    print(f"Car final angle {Gheading_estimate:.2f}")


def park(pca, sock, shared_race_mode, device):
    global Gheading_estimate, Gheading_start

    print(f">>> Parking start heading: {Gheading_estimate} {time.time()}")
    position = navigate(sock, narrow=False)
    dl = position['left_min_distance']
    dr = position['right_min_distance']

    if not Gclock_wise:
        stop_distance = 1.4 if dl > dr else 1.5
    else:
        stop_distance = 1.4 if dl > dr else 1.3
    # stop_distance = 1.5 if (Gclock_wise and dl < dr) or (not Gclock_wise and dl > dr) else 1.4  # 1.6,1.4

    print(f">>> Car alignment heading: {Gheading_estimate} {time.time()}")
    print(f"Front distance: {position['front_distance']:.2f}")
    print(f"stop_distance: {stop_distance:.2f}, left distance: {dl:.2f}, right distance: {dr:.2f}")
    first_line_led(device)
    align_parallel(pca, sock, shared_race_mode, stop_distance)
    time.sleep(5)

    #while True:
    #    position = navigate(sock,True)
    #    front_distance = position['front_distance']
    #    print(f"front_distance {front_distance} Gheading_estimate {Gheading_estimate}")

    print(f">>> Car turn heading: {Gheading_estimate}")
    second_line_led(device)
    align_angular(pca, sock, PARK_ANGLE if Gclock_wise else - PARK_ANGLE, shared_race_mode)
    print(f">>> Car final heading: {Gheading_estimate} {orientation(Gheading_estimate) - orientation(Gheading_start):.2f}")

    while True:
        position = navigate(sock)
        # print(f"Front distance: {position['front_distance']:.2f}")
        if position['front_distance'] < 0.05: break
        set_servo_angle(pca, 12, SERVO_BASIS)
        set_motor_speed(pca, 13, PARK_SPEED * MOTOR_FACTOR + MOTOR_BASIS)

    print("Stopping the vehicle, lifting rear axle ")
    set_motor_speed(pca, 13, MOTOR_BASIS)
    set_servo_angle(pca, 12, SERVO_BASIS)
    # set_servo_angle(pca, 11, 1.7)


def sensor_callback():
    global shared_race_mode, shared_blue_line_count
    if shared_race_mode.value == 0:
        print("Race started")
        shared_race_mode.value = 3


def get_clock_wise(sock):
    global Gline_orientation, Gclock_wise
    if Gline_orientation == "UP":
        print(f"Gline orientation: UP")
        Gclock_wise = False
    elif Gline_orientation == "DOWN":
        print(f"Gline orientation: DOWN")
        Gclock_wise = True
    else:
        position = navigate(sock)
        Gclock_wise = position['distance_ratio'] > 1.0
        print(f"distance_ratio: {position['distance_ratio']}")


def led_out(device, pattern):
    # Display the pattern on the LED matrix
    with canvas(device) as draw:
        for y, row in enumerate(pattern):
            for x in range(8):
                # Check if the specific bit in the row is set
                if (row >> (7 - x)) & 1:
                    draw.point((x, y), fill="white")
                else:
                    draw.point((x, y), fill="black")


def smiley_led(device):
    pattern = [
        0b00111100, 0b01000010, 0b10100101, 0b10000001,
        0b10100101, 0b10011001, 0b01000010, 0b00111100
    ]
    led_out(device, pattern)


def error_led(device):
    pattern = [
        0b11111111, 0b10000000, 0b10000000, 0b11111100,
        0b10000000, 0b10000000, 0b10000000, 0b11111111
    ]
    led_out(device, pattern)


def race_led(device):
    pattern = [
        0b11111100, 0b10000010, 0b10000010, 0b11111100,
        0b10100000, 0b10010000, 0b10001000, 0b10000100
    ]
    led_out(device, pattern)


def parking_led(device):
    pattern = [
        0b11111100, 0b10000010, 0b10000010, 0b11111100,
        0b10000000, 0b10000000, 0b10000000, 0b10000000
    ]
    led_out(device, pattern)


def second_line_led(device):
    pattern = [
        0b00111100, 0b01000010, 0b00000010, 0b00000100,
        0b00001000, 0b00010000, 0b00100000, 0b01111110
    ]
    led_out(device, pattern)


def first_line_led(device):
    pattern = [
        0b00011000, 0b00111000, 0b00011000, 0b00011000,
        0b00011000, 0b00011000, 0b01111110, 0b01111110
    ]
    led_out(device, pattern)


def blank_led(device):
    pattern = [
        0b00000000, 0b00000000, 0b00000000, 0b00000000,
        0b00000000, 0b00000000, 0b00000000, 0b00000000
    ]
    led_out(device, pattern)


# ID 2357:012e TP-Link 802.11ac NIC
def check_usb_device(vendor_id="2357", product_id="012e"):
    # Convert vendor_id and product_id from hexadecimal string to integer
    vendor_id = int(vendor_id, 16)
    product_id = int(product_id, 16)
    # Find USB device with specific Vendor ID and Product ID
    device = usb.core.find(idVendor=vendor_id, idProduct=product_id)
    # Return True if device is found, else False
    if device is not None:
        print("Device Found: ID {0}:{1}".format(vendor_id, product_id))
        return True
    else:
        return False


def main():
    global Gheading_estimate, Gheading_start, Gclock_wise
    global Gaccel_x, Gaccel_y, Gaccel_z, Gyaw
    global shared_race_mode
    global Glidar_moving_avg_fps, Gcamera_moving_avg_fps
    global Gobstacles

    print("Starting the UAV program...")
    Gobstacles = check_usb_device()
    if Gobstacles:
        print("Obstacle Race")
    else:
        print("Opening Race")

    # Create shared variables
    shared_GX = Value('d', 0.0)  # 'd' for double precision float
    shared_GY = Value('d', 0.0)

    # Initialize SPI connection and LED matrix device
    serial = spi(port=0, device=0, gpio=noop())
    device = max7219(serial, width=8, height=8)
    device.contrast(10)  # Adjust contrast if needed

    # Initialize touch button
    sensor = Button(SENSOR_PIN)
    sensor.when_pressed = sensor_callback

    # Initialize the I2C bus
    i2c = busio.I2C(SCL, SDA)

    # Create a simple PCA9685 class instance
    pca = PCA9685(i2c)
    pca.frequency = 50  # Standard servo frequency
    arm_esc(pca, 1)
    set_motor_speed(pca, 13, MOTOR_BASIS)
    set_servo_angle(pca, 12, SERVO_BASIS)
    set_servo_angle(pca, 11, 0.0)

    # gyro setup
    print("Initializing WT61 gyroscope sensor...")
    initialize_wt61()

    # LIDAR setup
    sock = connect_lidar()
    print('Getting LIDAR info...')
    sock, model, firmware_minor, firmware_major, hardware, serialnum_str = get_info(sock)
    print('LIDAR Info:', model, firmware_minor, firmware_major, hardware, serialnum_str)

    print('Getting LIDAR health...')
    health = get_health(sock)
    print('LIDAR Health:', health)

    print('Starting scan...')
    start_scan(sock)
    position = navigate(sock)
    print(f"Minimal distance {position['min_distance']:.2f} at angle {position['min_angle']:.2f}")
    print(f"Left minimal distance {position['left_min_distance']:.2f} at angle {position['left_min_angle']:.2f}")
    print(f"Right minimal distance {position['right_min_distance']:.2f} at angle {position['right_min_angle']:.2f}")
    print(f"Front distance {position['front_distance']:.2f}")

    # Camera setup
    picam0 = Picamera2(camera_num=0)
    picam1 = Picamera2(camera_num=1)
    config = {"format": 'RGB888', "size": (640, 400)}
    picam0.configure(picam0.create_preview_configuration(main=config))
    picam1.configure(picam1.create_preview_configuration(main=config))
    picam0.start()
    picam1.start()

    # Set camera controls to adjust exposure time and gain
    picam0.set_controls({"ExposureTime": 10000, "AnalogueGain": 10.0})
    picam1.set_controls({"ExposureTime": 10000, "AnalogueGain": 10.0})

    # Start threads and processes
    lidar_thread_instance = threading.Thread(target=lidar_thread,
                                             args=(sock, pca, shared_GX, shared_GY, shared_race_mode))
    camera_thread_instance = threading.Thread(target=camera_thread,
                                              args=(pca, picam0, picam1, shared_race_mode, device))
    gyro_thread_instance = threading.Thread(target=gyro_thread, args=(shared_race_mode,))

    xbox_controller_process_instance = Process(target=xbox_controller_process,
                                               args=(pca, shared_GX, shared_GY, shared_race_mode))

    lidar_thread_instance.start()
    camera_thread_instance.start()
    gyro_thread_instance.start()
    xbox_controller_process_instance.start()

    time.sleep(2)
    Gheading_start = Gheading_estimate
    print(f">>> All processes have started: {Gheading_start:.2f} degrees")
    print(f'LIDAR moving average FPS: {Glidar_moving_avg_fps:.2f}')
    print(f'Camera moving average FPS: {Gcamera_moving_avg_fps:.2f}')
    get_clock_wise(sock)
    print(f"Clockwise: {Gclock_wise}")
    smiley_led(device)

    print("Steering and power neutral.")
    set_motor_speed(pca, 13, MOTOR_BASIS)
    set_servo_angle(pca, 12, SERVO_BASIS)

    while True:
        position = navigate(sock,True)
        front_distance = position['front_distance']
        print(f"front_distance {front_distance} Gheading_estimate {Gheading_estimate} GPitch {GPitch} Groll {Groll}"")

    try:
        while (shared_race_mode.value != 3):
            time.sleep(0.1)

        start_time = time.time()
        shared_race_mode.value = 1
        race_led(device)

        while shared_race_mode.value != 2:
            time.sleep(0.1)

        # set_motor_speed(pca, 13, MOTOR_BASIS)
        # set_servo_angle(pca, 12, SERVO_BASIS)
        print(f"Race time: {time.time() - start_time:.2f} seconds")
        smiley_led(device)

        if PARKING_MODE:
            print(f"Time to start the parking procedure: {time.time() - start_time:.2f} seconds")
            parking_led(device)

            print("Starting the parking procedure")
            print(f">>> Car race end heading: {Gheading_estimate:.2f} {time.time()}")
            print(f"Heading estimate: {orientation(Gheading_estimate):.2f}")
            print(f"Heading start: {orientation(Gheading_start):.2f}")
            park(pca, sock, shared_race_mode, device)
            print(f"Race & parking time: {time.time() - start_time:.2f} seconds")

        set_motor_speed(pca, 13, MOTOR_BASIS)
        set_servo_angle(pca, 12, SERVO_BASIS)

        picam0.stop()
        picam1.stop()
        stop_scan(sock)
        sock.close()
        pygame.quit()
        sys.exit()

    except KeyboardInterrupt:
        picam0.stop()
        picam1.stop()
        stop_scan(sock)
        sock.close()
        pygame.quit()
        sys.exit()


if __name__ == '__main__':
    main()
