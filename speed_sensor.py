
# sudo apt-get install python3-gpiozero
# LM 393 Speed Sensor

from gpiozero import Button
import time

# Define the GPIO pin connected to the sensor
SENSOR_PIN = 17

# Initialize the pulse count
pulse_count = 0

def pulse_callback():
    global pulse_count
    pulse_count += 1

# Set up the sensor input
sensor = Button(SENSOR_PIN)

# Attach the callback function to the sensor's when_pressed event
sensor.when_pressed = pulse_callback

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
    print("Cleaning up GPIO")
