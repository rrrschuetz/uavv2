from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas
import time

def main():
    # Initialize SPI connection and LED matrix device
    serial = spi(port=0, device=0, gpio=noop())
    device = max7219(serial, width=8, height=8)
    device.contrast(10)  # Adjust contrast if needed

    # Define a simple pattern (smiley face) as an example
    smiley_pattern = [
        0b00111100,  # Row 1: ..####..
        0b01000010,  # Row 2: .#....#.
        0b10100101,  # Row 3: #.#..#.#
        0b10000001,  # Row 4: #......#
        0b10100101,  # Row 5: #.#..#.#
        0b10011001,  # Row 6: #..##..#
        0b01000010,  # Row 7: .#....#.
        0b00111100   # Row 8: ..####..
    ]

    # Display the pattern on the LED matrix
    while True:
        with canvas(device) as draw:
            for y, row in enumerate(smiley_pattern):
                for x in range(8):
                    # Check if the specific bit in the row is set
                    if (row >> (7 - x)) & 1:
                        draw.point((x, y), fill="white")
                    else:
                        draw.point((x, y), fill="black")
        time.sleep(1)  # Display for 1 second before updating

if __name__ == '__main__':
    main()
