from gpiozero import LED
import time

my_led = LED(17)
print("LED on")
my_led.on()
time.sleep(1)
print("LED off")
my_led.off()

