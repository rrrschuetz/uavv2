import socket
import struct
import serial
import pygame
import json, configparser
import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import trim_mean
import cv2
from picamera2 import Picamera2
from multiprocessing import Process, Value
import threading, multiprocessing
from collections import deque
from adafruit_pca9685 import PCA9685
import board
from board import SCL, SDA
import busio
from gpiozero import Button, LED
import qmc5883l as qmc5883
import time, sys, os
import usb.core, usb.util
import math, statistics
import torch

from sympy.codegen.ast import continue_
from torch.distributions.constraints import positive

from lidar_color_model import CNNModel  # Import the model from model.py
# from preprocessing import preprocess_input, load_scaler  # Import preprocessing functions
from preprocessing_no_scaler import preprocess_input  # Import preprocessing functions

config = configparser.ConfigParser()
config.read(os.path.expanduser('./config.ini'))

WRITE_CAMERA_IMAGE = config.getboolean('Race', 'WRITE_CAMERA_IMAGE')  # False
WRITE_CAMERA_MOVIE = config.getboolean('Race', 'WRITE_CAMERA_MOVIE')  # False
Gobstacles = not config.getboolean('Race', 'OPENING_RACE')  # False
TOTAL_LAPS = int(config['Race']['TOTAL_LAPS'])  # 3
PARKING_MODE = config.getboolean('Race', 'PARKING_MODE')  # True
LED_DISPLAY = config.getboolean('Race', 'LED_DISPLAY')  # False
READY_GESTURE = config.getboolean('Race', 'READY_GESTURE')  # False

SERVO_FACTOR = float(config['Steering']['SERVO_FACTOR'])  # 0.5  # 0.4
SERVO_BASIS = float(config['Steering']['SERVO_BASIS'])  # 0.5  # 1.55 # 0.55
MOTOR_FACTOR = float(config['Steering']['MOTOR_FACTOR'])  # 0.4 #0.45  # 0.3
MOTOR_FACTOR_OPENING = float(config['Steering']['MOTOR_FACTOR_OPENING'])  # 0.4 #0.45  # 0.3
MOTOR_BASIS = float(config['Steering']['MOTOR_BASIS'])  # 0.1
MOTOR_BOOST = float(config['Steering']['MOTOR_BOOST'])  # 0.2
MOTOR_ACCEL = float(config['Steering']['MOTOR_ACCEL'])  # 0.99
LIFTER_BASIS = float(config['Steering']['LIFTER_BASIS'])  # 1.45
LIFTER_UP = float(config['Steering']['LIFTER_UP'])  # 2.7

RACE_SPEED = float(config['Steering']['RACE_SPEED'])  # -0.35
EMERGENCY_SPEED = float(config['Steering']['EMERGENCY_SPEED'])  # -0.45
PARK_SPEED = float(config['Steering']['PARK_SPEED'])  # -0.3 #-0.35  # -0.55
PARK_STEER = float(config['Steering']['PARK_STEER'])  # 0.5   #2.5

CLOCKWISE_TURN_GREEN = float(config['Parking']['CLOCKWISE_TURN_GREEN'])
CLOCKWISE_TURN_RED = float(config['Parking']['CLOCKWISE_TURN_RED'])
COUNTERCLOCKWISE_TURN_GREEN = float(config['Parking']['COUNTERCLOCKWISE_TURN_GREEN'])
COUNTERCLOCKWISE_TURN_RED = float(config['Parking']['COUNTERCLOCKWISE_TURN_RED'])

# LED and LCD output
print(f"LED_DISPLAY {LED_DISPLAY}")
if LED_DISPLAY:
    from luma.led_matrix.device import max7219
    from luma.core.interface.serial import spi, noop
    from luma.core.render import canvas
else:
    from RPLCD.i2c import CharLCD

# Configuration for WT61 Gyroscope
SERIAL_PORT = "/dev/ttyAMA0"  # or "/dev/ttyS0" if you have mapped accordingly
BAUD_RATE = 115200
TIMEOUT = 0.5  # Set a slightly longer timeout to ensure full packet reads

SENSOR_PIN = 17

LIDAR_LEN = 1620
COLOR_LEN = 1280
ANGLE_CORRECTION = 180.0
DISTANCE_CORRECTION = -0.10

# Global variables
Gclock_wise = False
Gmodel_cc = None
Gmodel_cw = None
Glidar_string = ""
Gcolor_string = ",".join(["0"] * COLOR_LEN)
Gx_coords = np.zeros(COLOR_LEN, dtype=float)
Gline_orientation = None
Gfront_distance = 0.0
Gpitch = 0.0
Groll = 0.0
Gyaw = 0.0
Gaccel_x = 0.0
Gaccel_y = 0.0
Gaccel_z = 0.0
Gheading_estimate = 0.0
Gheading_start = 0.0
Glap_end = False
Gpca = None
Gboost = 0.0
Glidar_moving_avg_fps = 0.0
Gcamera_moving_avg_fps = 0.0
shared_race_mode = Value('i', 0)

i2c = board.I2C()
qmc = qmc5883.QMC5883L(i2c)
qmc.output_data_rate = (qmc5883.OUTPUT_DATA_RATE_200)


# Servo functions
def set_servo_angle(channel, angle):
    global Gpca
    pulse_min = 260  # Pulse width for 0 degrees
    pulse_max = 380  # Pulse width for 180 degrees
    pulse_width = pulse_min + angle * (pulse_max - pulse_min)
    try:
        Gpca.channels[channel].duty_cycle = int(pulse_width / 4096 * 0xFFFF)
    except ValueError:
        print("Invalid angle value: ", angle)


# ESC functions
def set_motor_speed(channel, speed):
    global Gpca
    pulse_min = 310  # Pulse width for 0% speed
    pulse_max = 409  # Pulse width for 100% speed
    pulse_width = pulse_min + speed * (pulse_max - pulse_min)
    try:
        Gpca.channels[channel].duty_cycle = int(pulse_width / 4096 * 0xFFFF)
    except ValueError:
        print("Invalid speed value: ", speed)


def arm_esc(channel):
    print("Arming ESC...")
    set_motor_speed(channel, 0)
    time.sleep(1)
    print("ESC armed")


def start_boost(boost):
    global Gboost
    Gboost = boost

    # short set back
    set_motor_speed(13, - RACE_SPEED * MOTOR_FACTOR + MOTOR_BASIS)
    time.sleep(0.2)
    set_motor_speed(13, MOTOR_BASIS)

    # Check acclelaration every 0.1 sec
    threading.Timer(0.02, stop_boost).start()
    print(f"Booster of {Gboost} activated.")


def stop_boost():
    global Gboost, Gaccel_x, Gaccel_y, Gaccel_z
    accel = math.sqrt(Gaccel_x ** 2 + Gaccel_y ** 2 + Gaccel_z ** 2)
    print(f"Acceleration: Minimum {MOTOR_ACCEL} x/y/z/a {Gaccel_x}/{Gaccel_y}/{Gaccel_z}/{accel}")
    if accel < MOTOR_ACCEL:
        Gboost *= 1.01  # escalate if uav is not moving
        start_boost(Gboost)
        print("Booster reactivated.")
    else:
        Gboost = 0.0
        print("Booster deactivated.")


# LIDAR functions
IP_ADDRESS = '192.168.11.2'
PORT = 8089


def connect_lidar(ip=IP_ADDRESS, port=PORT):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect((ip, port))
    return sock


def receive_full_data(sock, expected_length, timeout=0.1):  # 5
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


def lidar_thread(sock, shared_GX, shared_GY, shared_race_mode, stop_event):
    global Glidar_string, Gcolor_string
    global Gx_coords, Gfront_distance
    global Glidar_moving_avg_fps
    global Gobstacles, Gboost
    global Gmodel_cc, Gmodel_cw

    window_size = 10
    input_size = 100  # not too wide !

    # scaler_lidar = None
    device = None

    fps_list = deque(maxlen=10)
    while not stop_event.is_set():
        start_time = time.time()

        if shared_race_mode.value == 4:  # Training
            # print("LIDAR in manual mode")
            interpolated_distances, angles = full_scan(sock)
            Glidar_string = ",".join(f"{d:.4f}" for d in interpolated_distances[:LIDAR_LEN])
            with open("data_file.txt", "a") as file:
                file.write(f"{shared_GX.value},{shared_GY.value},{Glidar_string},{Gcolor_string}\n")

        elif shared_race_mode.value in [1, 3]:  # Race
            # print("LIDAR in autonomous mode")
            interpolated_distances, angles = full_scan(sock)

            valid_distances = median_filter(interpolated_distances[:LIDAR_LEN], size=window_size)
            Gfront_distance = np.max(valid_distances[LIDAR_LEN // 2 - input_size // 2:LIDAR_LEN // 2 + input_size // 2])

            if Gmodel_cc is None:
                # Load the trained model and the scaler
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # Initialize the model
                Gmodel_cc = CNNModel(LIDAR_LEN, COLOR_LEN).to(device)
                Gmodel_cw = CNNModel(LIDAR_LEN, COLOR_LEN).to(device)

                # Load the trained weights into the model
                if Gobstacles:
                    if shared_race_mode.value == 1:
                        state_dict_cc = torch.load('./model_cc.pth', map_location=torch.device('cpu'))
                        state_dict_cw = torch.load('./model_cw.pth', map_location=torch.device('cpu'))
                    else:
                        state_dict_cc = torch.load('./model_cc_park.pth', map_location=torch.device('cpu'))
                        state_dict_cw = torch.load('./model_cw_park.pth', map_location=torch.device('cpu'))
                else:
                    state_dict_cc = torch.load('./model_cc_opening.pth', map_location=torch.device('cpu'))
                    state_dict_cw = torch.load('./model_cw_opening.pth', map_location=torch.device('cpu'))

                # Convert all weights to float32 if they are in float64
                for key, value in state_dict_cc.items():
                    if value.dtype == torch.float64:  # Check if the parameter is in double precision
                        state_dict_cc[key] = value.float()  # Convert to single precision (float32)
                for key, value in state_dict_cw.items():
                    if value.dtype == torch.float64:  # Check if the parameter is in double precision
                        state_dict_cw[key] = value.float()  # Convert to single precision (float32)

                # Load the state dict into the model
                Gmodel_cc.load_state_dict(state_dict_cc)
                Gmodel_cw.load_state_dict(state_dict_cw)
                Gmodel_cc.eval()
                Gmodel_cw.eval()

            # if scaler_lidar is None:
            #    # Load the scaler for LIDAR data
            #    scaler_lidar = load_scaler('./scaler.pkl')

            # emergency break
            left_min = min(interpolated_distances[LIDAR_LEN // 8 * 3: LIDAR_LEN // 2])
            right_min = min(interpolated_distances[LIDAR_LEN // 2: LIDAR_LEN // 8 * 5])
            if 0 < left_min < 0.07 or 0 < right_min < 0.07:
                if shared_race_mode.value == 1:
                    print("Emergency break")
                    dir = 1 if left_min < right_min else -1
                    set_servo_angle(12, SERVO_BASIS + PARK_STEER * SERVO_FACTOR * dir)
                    time.sleep(0.5)
                    set_motor_speed(13, MOTOR_BASIS - EMERGENCY_SPEED * MOTOR_FACTOR)
                    time.sleep(0.5)
                    set_motor_speed(13, MOTOR_BASIS)
                    set_servo_angle(12, SERVO_BASIS)
                    time.sleep(0.5)
                elif shared_race_mode.value == 3:
                    shared_race_mode.value = 6
                    set_motor_speed(13, MOTOR_BASIS)
                    set_servo_angle(12, SERVO_BASIS)
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
                        output = Gmodel_cc(lidar_tensor, color_tensor)
                    else:
                        output = Gmodel_cw(lidar_tensor, color_tensor)

                # Convert the model's output to steering commands or other UAV controls
                steering_commands = output.cpu().numpy()
                # print("Steering Commands:", steering_commands)
                X = steering_commands[0, 0]  # Extract GX (first element of the output)
                # Y = steering_commands[0, 1]  # Extract GY (second element of the output)
                Y = RACE_SPEED
                # if Gclock_wise:
                #    X = -X
                motor = Gboost + MOTOR_FACTOR if Gobstacles else MOTOR_FACTOR_OPENING
                if -1.0 < X < 1.0 and -1.0 < Y < 0.0:
                    set_servo_angle(12, X * SERVO_FACTOR + SERVO_BASIS)
                    set_motor_speed(13, Y * motor + MOTOR_BASIS)
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
class uav_cam(Picamera2):
    def __init__(self, camera_num):
        super().__init__(camera_num=camera_num)
        self.camera_num = camera_num
        self.config = self.create_still_configuration(main={"format": 'RGB888', "size": (640, 480)})
        self.configure(self.config)
        self.start()
        time.sleep(2)

        # Automatischer AWB zum Kalibrieren
        image_auto = self.capture_array()
        r, g, b = self._get_mean_rgb(image_auto)
        self.r_gain, self.b_gain = self._compute_awb_gains(r, g, b)
        print(f"[INFO] Camera {self.camera_num} AWB-Gains gesetzt: R={self.r_gain}, B={self.b_gain}")

        # Manuellen Weißabgleich setzen
        print("Max saturation set.")
        self.set_controls({
            "Saturation": 2.0,
            "AwbEnable": True,
            "ColourGains": (self.r_gain, self.b_gain)
        })
        time.sleep(1)

    def image(self):
        image = self.capture_array()
        image = cv2.flip(image, -1)
        image = self._gamma_correction(image)
        image = self._enhance_lighting(image)
        return image  # cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # ---------- Bildverbesserung ----------
    def _gamma_correction(self, image, gamma=1.5):
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)

    #def _enhance_lighting(self, image):
    #    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #    h, s, v = cv2.split(hsv)
    #    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #    v = clahe.apply(v)
    #    hsv = cv2.merge([h, s, v])
    #    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _enhance_lighting(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # ---------- Weißabgleich ----------
    def _get_mean_rgb(self, image):
        h, w, _ = image.shape
        center = image[h // 3:2 * h // 3, w // 3:2 * w // 3]
        mean = cv2.mean(center)[:3]  # BGR
        # return mean[2], mean[1], mean[0]  # R, G, B
        return mean[0], mean[1], mean[2]  # R, G, B

    def _compute_awb_gains(self, r, g, b):
        avg = (r + g + b) / 3
        r_gain = avg / r if r != 0 else 1.0
        b_gain = avg / b if b != 0 else 1.0
        return round(r_gain, 2), round(b_gain, 2)


class mask():

    #def red(self, image):
    #    lower_red1 = np.array([0, 50, 50])
    #    upper_red1 = np.array([10, 255, 255])
    #    lower_red2 = np.array([160, 50, 50])
    #    upper_red2 = np.array([180, 255, 255])
    #    red_mask = cv2.inRange(image, lower_red1, upper_red1) | cv2.inRange(image, lower_red2, upper_red2)
    #    red_mask = self._remove_small_contours(self._apply_morphological_operations(red_mask))
    #    return red_mask

    def red(self, image):
        lower_red = np.array([0, 160, 0])  #0,140,50  #0,150,80  #0,130,0 #20.150.140
        upper_red = np.array([255, 255, 255])  #255,255,210 #255,255,170 #255,255,255 #255,200,200
        red_mask = cv2.inRange(image, lower_red, upper_red)
        return red_mask

    #def green(self, image):
    #    lower_green = np.array([35, 40, 40])
    #    upper_green = np.array([90, 255, 255])
    #    green_mask = cv2.inRange(image, lower_green, upper_green)
    #    green_mask = self._remove_small_contours(self._apply_morphological_operations(green_mask))
    #    return green_mask

    def green(self, image):
        lower_green = np.array([0, 0, 0]) #0,0,50 #0,0,120 #0,0,120 #20,0,0
        upper_green = np.array([255, 100, 255])  #255,120,255 #255,100,255
        green_mask = cv2.inRange(image, lower_green, upper_green)
        return green_mask

    #def blue(self, image):
    #    blue_lower = np.array([90, 70, 90])  # HSV range for blue detection
    #    blue_upper = np.array([140, 255, 255])
    #    blue_mask = cv2.inRange(image, blue_lower, blue_upper)
    #    blue_mask = self._remove_small_contours(blue_mask)
    #    return blue_mask

    def blue(self, image):
        lower_blue = np.array([0, 0, 0])
        upper_blue = np.array([255, 255, 100])
        blue_mask = cv2.inRange(image, lower_blue, upper_blue)
        return blue_mask

    #def amber(self, image):
    #    amber_lower = np.array([10, 50, 50])  # Lower bound for hue, saturation, and brightness
    #    amber_upper = np.array([20, 255, 255])  # Upper bound for hue, saturation, and brightness
    #    amber_mask = cv2.inRange(image, amber_lower, amber_upper)
    #    amber_mask = self._remove_small_contours(amber_mask)
    #    return amber_mask

    def amber(self, image):
        lower_amber = np.array([20, 120, 150])
        upper_amber = np.array([255, 200, 255])
        amber_mask = cv2.inRange(image, lower_amber, upper_amber)
        return amber_mask

    #def magenta(self, image):
    #    magenta_lower = np.array([140, 50, 50])  # HSV range for magenta color detection
    #    magenta_upper = np.array([170, 255, 255])
    #    magenta_mask = cv2.inRange(image, magenta_lower, magenta_upper)
    #    magenta_mask = self._remove_small_contours(self._apply_morphological_operations(magenta_mask))
    #    return magenta_mask

    def magenta(self, image):
        lower_magenta = np.array([0, 160, 0])   #20,150,150
        upper_magenta = np.array([255, 255, 127])
        magenta_mask = cv2.inRange(image, lower_magenta, upper_magenta)
        return magenta_mask

    # ---------- Maskenfilterung ----------
    def _apply_morphological_operations(self, mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=4)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)
        return mask

    def _remove_small_contours(self, mask, min_area=500):
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
        if shared_race_mode.value == 3 or ( \
            aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1] and
            angle_range[0] <= angle <= angle_range[1]):
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
    mask_filter = mask()

    #hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #red_mask = mask_filter.red(hsv_image)
    #green_mask = mask_filter.green(hsv_image)

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    red_mask = mask_filter.red(lab_image)
    green_mask = mask_filter.green(lab_image)
    magenta_mask = mask_filter.magenta(lab_image)

    combined_mask = cv2.bitwise_or(red_mask, green_mask)
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = filter_contours(contours)

    #x_coords = np.zeros(hsv_image.shape[1], dtype=float)
    x_coords = np.zeros(lab_image.shape[1], dtype=float)

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
        cv2.drawContours(image, [box], -1, (255, 255, 255), 3)
        cv2.putText(image, label, center, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2)

    if (num_detector_calls % 2 == 0):

        # Detect blue and amber lines
        if Gclock_wise:
            #first_line_mask = mask_filter.amber(hsv_image)
            #second_line_mask = mask_filter.blue(hsv_image)
            first_line_mask = mask_filter.amber(lab_image)
            second_line_mask = mask_filter.blue(lab_image)
        else:
            #first_line_mask = mask_filter.blue(hsv_image)
            #second_line_mask = mask_filter.amber(hsv_image)
            first_line_mask = mask_filter.blue(lab_image)
            second_line_mask = mask_filter.amber(lab_image)

        most_significant_line = None
        max_line_length = 0
        #height, width = hsv_image.shape[:2]
        height, width = lab_image.shape[:2]

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

        # Find and filter contours for magenta blobs only with outwards looking camera
        #contours, _ = cv2.findContours(mask_filter.magenta(hsv_image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(magenta_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # 5000 size of parking lot
                rect = cv2.minAreaRect(contour)
                center = (int(rect[0][0]), int(rect[0][1]))
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                left_end = min(box[:, 0])
                right_end = max(box[:, 0])
                #print(f"Magenta rectangle detected: {area} pixels")
                magenta_rectangle = True
                cv2.drawContours(image, [contour], -1, (255, 255, 255), 3)  # Draw the magenta rectangle
                cv2.putText(image, "M", center, cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 2)
        # Add timestamp in the lower left corner
        timestamp = time.strftime("%H:%M:%S", time.localtime()) + f":{int((time.time() % 1) * 100):02d}"
        cv2.putText(image, timestamp, (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if WRITE_CAMERA_IMAGE and num_detector_calls % 2 == 0:
        cv2.imwrite("labeled_image.jpg", image)
        # cv2.imwrite('amber_mask.jpg', amber_mask)
        # cv2.imwrite('blue_mask.jpg', blue_mask)
        # cv2.imwrite('magenta_mask.jpg', magenta_mask)
        # cv2.imwrite('red_mask.jpg', red_mask)
        # cv2.imwrite('green_mask.jpg', green_mask)

    return x_coords, first_line, second_line, magenta_rectangle, line_orientation, image, red_mask, green_mask, magenta_mask


def camera_thread(uav_camera0, uav_camera1, shared_race_mode, device, stop_event):
    global Gcolor_string, Gx_coords, Gfront_distance
    global Gline_orientation
    global Glap_end, Gheading_estimate  # heading
    global Gcamera_moving_avg_fps, Glidar_moving_avg_fps

    fps_list = deque(maxlen=10)
    frame_height, frame_width, _ = uav_camera0.capture_array().shape
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
    heading_prev_lap = Gheading_estimate
    max_heading = 0
    cum_heading = 0

    # detector = MarkingDetector(output_dir="detections")

    try:
        while not stop_event.is_set():
            num_detector_calls += 1
            if shared_race_mode.value in [0, 1, 3, 4]:
                start_time = time.time()

                cum_heading += yaw_difference(heading_prev_lap, Gheading_estimate)
                heading_prev_lap = Gheading_estimate
                # print("cum_heading ",cum_heading)

                image0 = uav_camera0.image()
                image1 = uav_camera1.image()
                image = np.hstack((image0, image1))
                image = image[frame_height:, :]

                Gx_coords, first_line, second_line, parking_lot, line_orientation, image, red_mask, green_mask, magenta_mask \
                    = detect_and_label_blobs(image, num_detector_calls)

                # if Gclock_wise:
                #    Gx_coords = Gx_coords * -1.0
                #    Gx_coords = Gx_coords[::-1]
                Gcolor_string = ",".join(map(str, Gx_coords.astype(int)))

                if first_line and Gline_orientation is None: Gline_orientation = line_orientation

                # Avoid mix of red obstacles and magenta parking lot in curves
                # if orientation(yaw_difference(Gheading_start, Gheading_estimate)) > 10: parking_lot = False

                if shared_race_mode.value == 1:

                    # print(f"max_heading {max_heading} Gheading_estimate {Gheading_estimate}")
                    max_heading = max(max_heading, Gheading_estimate)
                    #if Glap_end and max_heading > 350 and abs(cum_heading) > 100 and Gfront_distance < 1.3:
                    if max_heading > 350 and abs(cum_heading) > 290:
                        print(f"max_heading {max_heading} cum_heading {cum_heading} Gfront_distance {Gfront_distance}")
                        num_laps += 1
                        max_heading = 0
                        cum_heading = 0
                        print(f"Laps completed: {num_laps} / {Gheading_estimate:.2f}")
                        # print(f'LIDAR moving average FPS: {Glidar_moving_avg_fps:.2f}')
                        # print(f'Camera moving average FPS: {Gcamera_moving_avg_fps:.2f}')
                        if num_laps >= TOTAL_LAPS:
                            shared_race_mode.value = 2
                            print("End of race.")

                elif shared_race_mode.value == 3:
                    #shared_race_mode.value = 6
                    print("Waiting for parking end")

                # Save the image with labeled contours
                if WRITE_CAMERA_MOVIE:
                    # Farbliche Masken erzeugen
                    red_colored = cv2.merge([red_mask, np.zeros_like(red_mask), np.zeros_like(red_mask)])
                    green_colored = cv2.merge([np.zeros_like(green_mask), green_mask, np.zeros_like(green_mask)])
                    magenta_colored = cv2.merge([np.zeros_like(magenta_mask), magenta_mask, np.zeros_like(magenta_mask)])

                    # Originalbild overlayen
                    overlayed_image = cv2.addWeighted(image, 0.5, red_colored, 0.5, 0)
                    overlayed_image = cv2.addWeighted(overlayed_image, 0.5, green_colored, 0.5, 0)
                    overlayed_image = cv2.addWeighted(overlayed_image, 0.5, magenta_colored, 0.5, 0)
                    video_writer.write(overlayed_image)
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


def xbox_controller_process(shared_GX, shared_GY, shared_race_mode, stop_event):
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

    while not stop_event.is_set():
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                # print(f"JOYAXISMOTION: axis={event.axis}, value={event.value}")
                if event.axis == 1:
                    shared_GY.value = event.value
                    set_motor_speed(13, event.value * MOTOR_FACTOR + MOTOR_BASIS)
                elif event.axis == 2:
                    shared_GX.value = event.value
                    set_servo_angle(12, event.value * SERVO_FACTOR + SERVO_BASIS)
                elif event.axis == 3:
                    pass
                    # set_servo_angle( 11, abs(event.value) * 1.5 + 0.0)

            elif event.type == pygame.JOYBALLMOTION:
                print(f"JOYBALLMOTION: ball={event.ball}, rel={event.rel}")

            elif event.type == pygame.JOYBUTTONDOWN:
                print(f"JOYBUTTONDOWN: button={event.button}")
                if event.button == 0:  # A button
                    if shared_race_mode.value == 0:
                        print("Race started")
                        shared_race_mode.value = 1
                elif event.button == 1:  # B button
                    if shared_race_mode.value == 0:
                        print("Training started")
                        shared_race_mode.value = 4
                elif event.button == 3:  # X button
                    print("STOP")
                    shared_race_mode.value = 0
                    set_motor_speed(13, MOTOR_BASIS)
                    set_servo_angle(12, SERVO_BASIS)
                elif event.button == 4:  # Y button
                    print("Training terminated")
                    shared_race_mode.value = 5
                    set_motor_speed(13, MOTOR_BASIS)
                    set_servo_angle(12, SERVO_BASIS)

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


Gcalibration_cache = None


def load_compass_calibration(filename="compass_calibration.json"):
    global Gcalibration_cache
    if Gcalibration_cache is None:
        with open(filename, "r") as f:
            d = json.load(f)
        Gcalibration_cache = (d["offsets"], d["scales"])
    return Gcalibration_cache


def get_magnetometer_heading():
    offsets, scales = load_compass_calibration()
    raw_x, raw_y, raw_z = qmc.magnetic
    # Hard-Iron + Soft-Iron
    x_corr = (raw_x - offsets[0]) * scales[0]
    y_corr = (raw_y - offsets[1]) * scales[1]
    heading = (math.degrees(math.atan2(y_corr, x_corr)) + 360) % 360
    return heading


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


def gyro_thread(shared_race_mode, stop_event):
    global Gaccel_x, Gaccel_y, Gaccel_z
    global Gpitch, Groll, Gyaw
    global Gheading_estimate, Gheading_start, Glap_end

    buff = bytearray()  # Buffer to store incoming serial data
    packet_counter = 0  # Counter to skip packets

    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT) as ser:
            ser.reset_input_buffer()  # Clear old data at the start

            while not stop_event.is_set():
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
                    # print(f"... Gheading_estimate: {Gheading_estimate:.2f}")
                    Glap_end = yaw_diff < 10  # no larger value !!
                    time.sleep(0.02)

    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except KeyboardInterrupt:
        print("Stopping data read.")


def park():
    print("Stopping the vehicle, lifting rear axle ")
    set_motor_speed(13, PARK_SPEED * MOTOR_FACTOR + MOTOR_BASIS)
    set_servo_angle(12, SERVO_BASIS)
    time.sleep(1)
    set_motor_speed(13, MOTOR_BASIS)
    set_servo_angle(12, SERVO_BASIS)
    set_servo_angle(11, LIFTER_UP)


def sensor_callback():
    global shared_race_mode
    if shared_race_mode.value in [0, 2]:
        print("Race started")
        start_boost(MOTOR_BOOST)
        shared_race_mode.value = 1


def get_clock_wise(sock):
    global Gline_orientation, Gclock_wise, Gobstacles

    if Gline_orientation == "UP":
        print(f"Gline orientation: UP")
        Gclock_wise = False
    elif Gline_orientation == "DOWN":
        print(f"Gline orientation: DOWN")
        Gclock_wise = True
    else:
        position = navigate(sock)
        dq = position['distance_ratio']
        dl = position['left_min_distance']
        dr = position['right_min_distance']
        Gclock_wise = dq > 1.0
        if not Gobstacles:
            if dr < 0.1:
                Gclock_wise = True
            elif dl < 0.1:
                Gclock_wise = False
        print(f"distance_ratio: {dq} left distance {dl} right distance {dr}")


class LCD():
    def __init__(self):
        # self.lcd = CharLCD('PCF8574', 0x27)
        self.display_buffer = []

    def display_message(self, message):
        return
        self.display_buffer.append(message)
        # Keep only the last 2 lines for LCD1602
        self.display_buffer = self.display_buffer[-2:]
        self.lcd.clear()
        for line in self.display_buffer:
            self.lcd.write_string(line)
            self.lcd.crlf()


def lcd_out(device, message):
    device.display_message(message)


def led_out(device, pattern):
    if not LED_DISPLAY: return
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


def ready_gesture():
    for i in range(2):
        set_servo_angle(12, SERVO_BASIS + SERVO_FACTOR / 2)
        time.sleep(1)
        set_servo_angle(12, SERVO_BASIS - SERVO_FACTOR / 2)
        time.sleep(1)
    set_servo_angle(12, SERVO_BASIS)


def ready_led():
    led = LED(4)
    led.on()
    time.sleep(3)
    led.off()


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
    global Gmodel_cc, Gmodel_cw, Gpca

    print("Starting the UAV program...")
    # Gobstacles = check_usb_device()
    if Gobstacles:
        print("Obstacle Race")
    else:
        print("Opening Race")

    # Create shared variables
    shared_GX = Value('d', 0.0)  # 'd' for double precision float
    shared_GY = Value('d', 0.0)

    # Initialize SPI connection and LED matrix
    device = None
    if LED_DISPLAY:
        print("LED display instance defined")
        serial = spi(port=0, device=0, gpio=noop())
        device = max7219(serial, width=8, height=8)
        device.contrast(10)  # Adjust contrast if needed
    else:
        print("LCD display instance defined")
        device = LCD()

    # Initialize touch button
    sensor = Button(SENSOR_PIN)
    sensor.when_pressed = sensor_callback

    # Initialize the I2C bus
    i2c = busio.I2C(SCL, SDA)

    # Create a simple PCA9685 class instance
    Gpca = PCA9685(i2c)
    Gpca.frequency = 50  # Standard servo frequency
    arm_esc(1)
    set_motor_speed(13, MOTOR_BASIS)
    set_servo_angle(12, SERVO_BASIS)
    set_servo_angle(11, LIFTER_BASIS)  # Lifter neutral

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
    print("Camera setup ...")
    picam0 = uav_cam(camera_num=0)
    picam1 = uav_cam(camera_num=1)

    # Start threads and processes
    print("Preparing processes ...")
    stop_event = threading.Event()
    lidar_thread_instance = threading.Thread(target=lidar_thread,
                                             args=(sock, shared_GX, shared_GY, shared_race_mode, stop_event))
    camera_thread_instance = threading.Thread(target=camera_thread,
                                              args=(picam0, picam1, shared_race_mode, device, stop_event))
    gyro_thread_instance = threading.Thread(target=gyro_thread, args=(shared_race_mode, stop_event))

    proc_stop = multiprocessing.Event()
    xbox_controller_process_instance = Process(target=xbox_controller_process,
                                               args=(shared_GX, shared_GY, shared_race_mode, proc_stop))

    print("Starting processes ...")
    print("LIDAR ...")
    lidar_thread_instance.start()
    print("Cameras ...")
    camera_thread_instance.start()
    print("Gyro and compass ...")
    gyro_thread_instance.start()
    print("Remote controller for training ...")
    xbox_controller_process_instance.start()
    print("Passed.")

    time.sleep(2)
    Gheading_start = Gheading_estimate
    print(f">>> All processes have started: {Gheading_start:.2f} degrees")
    print(f'LIDAR moving average FPS: {Glidar_moving_avg_fps:.2f}')
    print(f'Camera moving average FPS: {Gcamera_moving_avg_fps:.2f}')
    get_clock_wise(sock)
    print(f"Clockwise: {Gclock_wise}")

    # Show readiness for race
    smiley_led(device)
    lcd_out(device, f"READY - Clockwise: {Gclock_wise}")
    if READY_GESTURE: ready_gesture()
    ready_led()

    print("Steering and power neutral.")
    set_motor_speed(13, MOTOR_BASIS)
    set_servo_angle(12, SERVO_BASIS)


    try:
        while shared_race_mode.value == 0:
            time.sleep(0.1)

        if shared_race_mode.value == 1:  # Race

            start_time = time.time()
            set_motor_speed(13, MOTOR_BASIS)
            set_servo_angle(12, SERVO_BASIS)
            while shared_race_mode.value != 2:
                time.sleep(0.1)

            for _ in range(10):
                set_motor_speed(13, MOTOR_BASIS)
                time.sleep(0.1)

            print(f"Race time: {time.time() - start_time:.2f} seconds")
            smiley_led(device)

            if PARKING_MODE and Gobstacles:
                Gmodel_cc = None
                Gmodel_cw = None
                time.sleep(3)
                shared_race_mode.value = 3
                start_boost(MOTOR_BOOST)
                while shared_race_mode.value != 6:
                    time.sleep(0.1)
                park()
                time.sleep(10)
                set_servo_angle(11, LIFTER_BASIS)
                time.sleep(2)

            shared_race_mode.value = 5  # Termination

        else:  # Training

            while shared_race_mode.value != 5:  # End of training
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("Program interrupted.")

    finally:
        print("Shutdown.")
        stop_event.set()  # Threads stoppen
        proc_stop.set()  # Process stoppen
        lidar_thread_instance.join()
        camera_thread_instance.join()
        gyro_thread_instance.join()
        xbox_controller_process_instance.join()
        print("All threads and processes stopped cleanly")

        print("Stopping camera 0")
        picam0.stop()
        print("Stopping camera 1")
        picam1.stop()
        print("Stopping LIDAR")
        stop_scan(sock)
        sock.close()
        print("Stopping remote control")
        pygame.quit()
        print("Exiting")
        sys.exit(0)


if __name__ == '__main__':
    main()
