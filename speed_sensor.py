
# sudo apt-get install python3-rpi.gpio
# LM 393 Speed Sensor

import RPi.GPIO as GPIO
import time

# Define the GPIO pin connected to the sensor
SENSOR_PIN = 17

# Initialize the pulse count
pulse_count = 0

def pulse_callback(channel):
    global pulse_count
    pulse_count += 1

# Set up the GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(SENSOR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Add an interrupt to detect pulses
GPIO.add_event_detect(SENSOR_PIN, GPIO.FALLING, callback=pulse_callback)

try:
    start_time = time.time()
    while True:
        time.sleep(1)
        elapsed_time = time.time() - start_time
        # Calculate the speed (pulses per second)
        speed = pulse_count / elapsed_time
        print(f"Speed: {speed:.2f} pulses per second")
        pulse_count = 0
        start_time = time.time()
except KeyboardInterrupt:
    print("Program stopped by User")
finally:
    GPIO.cleanup()
