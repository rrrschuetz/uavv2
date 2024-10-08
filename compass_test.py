import machine
import sys
import math
from time import sleep

pinSDA = machine.Pin(20)
pinSCL = machine.Pin(21)
QMC5883L_ADDR = 0x0D
i2c = machine.I2C(0, freq=2000000, scl=pinSCL, sda=pinSDA)
devices = i2c.scan()

if not (QMC5883L_ADDR in devices):
    print("Not found GY-271 (QMC5883L)!")
    sys.exit(1)

############## Control Registers
RegCTRL1 = 0x09  # Control Register--> MSB(OSR:2,RNG:2,ODR:2,MODE:2)LSB
RegCTRL2 = 0x0A  # Control Register2--> MSB(Soft_RS:1,Rol_PNT:1,none:5,INT_ENB:1)LSB
RegFBR = 0x0B  # SET/RESET Period Register--> MSB(FBR:8)LSB

############## Control Register Value 
Mode_Standby = 0b00000000
Mode_Continuous = 0b00000001
ODR_10Hz = 0b00000000
ODR_50Hz = 0b00000100
ODR_100Hz = 0b00001000
ODR_200Hz = 0b00001100
RNG_2G = 0b00000000
RNG_8G = 0b00010000
OSR_512 = 0b00000000
OSR_256 = 0b01000000
OSR_128 = 0b10000000
OSR_64 = 0b11000000


def twos_comp(val, bits):
    """compute the 2's complement of int value val"""
    if (val & (1 << (bits - 1))) != 0:  # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)  # compute negative value
    return val  # return positive value as is


def get_bearing_raw(x, y):
    """Horizontal bearing (in degrees) from magnetic value X and Y."""
    if x is None or y is None:
        return None
    else:
        b = math.degrees(math.atan2(y, x))
        if b < 0:
            b += 360
        return round(b)


########### Init
ctrl1 = bytearray([Mode_Continuous | ODR_100Hz | RNG_2G | OSR_512])
i2c.writeto_mem(QMC5883L_ADDR, RegCTRL1, ctrl1)
i2c.writeto_mem(QMC5883L_ADDR, RegFBR, b'\x01')

while True:
    ########### Read
    buffer = i2c.readfrom_mem(QMC5883L_ADDR, 0, 14)
    xLo = buffer[0]
    xHi = buffer[1] << 8
    yLo = buffer[2]
    yHi = buffer[3] << 8
    zLo = buffer[4]
    zHi = buffer[5] << 8
    x = bin(xLo + xHi)[2:]
    y = bin(yLo + yHi)[2:]
    z = bin(zLo + zHi)[2:]
    compass_status = buffer[6]
    tLo = buffer[7]  # LSD
    tHi = buffer[8] << 8  # MSD
    compass_mode = buffer[9]
    chipID = buffer[13]

    ########### Convert
    ##### raw magnetic
    x_raw = twos_comp(int(x, 2), len(x))
    y_raw = twos_comp(int(y, 2), len(y))
    z_raw = twos_comp(int(z, 2), len(z))

    ########### Temperature
    T = bin(tLo + tHi)[2:]
    temperature = round((((twos_comp(int(T, 2), len(T))) / 100) + 37.6), 2)

    ########### Show result
    print(f'Bearing: {get_bearing_raw(x_raw, y_raw)}{chr(176)}  Temperature: {"%.1f" % temperature}{chr(176)}C')

    sleep(0.25)

