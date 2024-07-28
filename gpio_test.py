from gpiozero import LED, Button
from time import sleep

# Define GPIO pins for testing
test_pins = [
    {"output": 17, "input": 18},
    # Add more pairs as needed
]

def test_gpio_pair(output_pin, input_pin):
    """
    Test a pair of GPIO pins where one pin is set as output and the other as input.
    """
    # Setup the output and input pins
    output = LED(output_pin)
    input = Button(input_pin)

    # Test output HIGH
    output.on()
    sleep(0.1)
    if not input.is_pressed:
        print(f"Test failed: GPIO {output_pin} -> GPIO {input_pin} (HIGH)")
        return False

    # Test output LOW
    output.off()
    sleep(0.1)
    if input.is_pressed:
        print(f"Test failed: GPIO {output_pin} -> GPIO {input_pin} (LOW)")
        return False

    print(f"Test passed: GPIO {output_pin} -> GPIO {input_pin}")
    return True

def main():
    all_tests_passed = True

    for pins in test_pins:
        output_pin = pins["output"]
        input_pin = pins["input"]
        print(f"Testing GPIO {output_pin} -> GPIO {input_pin}")
        if not test_gpio_pair(output_pin, input_pin):
            all_tests_passed = False

    if all_tests_passed:
        print("All GPIO tests passed!")
    else:
        print("Some GPIO tests failed.")

if __name__ == "__main__":
    main()
