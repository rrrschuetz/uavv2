from picamera2 import Picamera2, Preview
import cv2
import numpy as np

def main():
    # Initialize the camera
    picam2 = Picamera2()

    # Configure the camera
    picam2.configure(picam2.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)}))

    # Start the camera
    picam2.start()

    while True:
        # Capture an image
        image = picam2.capture_array()

        # Display the image
        cv2.imshow('Camera', image)

        # Press 'q' on the keyboard to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera
    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
