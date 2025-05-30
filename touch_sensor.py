
from gpiozero import Button
import time

# Define the GPIO pin connected to the sensor
SENSOR_PIN = 17

def pulse_callback():
    print ("Pulse detected")

# Set up the sensor input
sensor = Button(SENSOR_PIN)

# Attach the callback function to the sensor's when_pressed event
sensor.when_pressed = pulse_callback

try:
    while True:
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Program stopped by User")
finally:
    print("Cleaning up GPIO")
