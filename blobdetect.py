import cv2
import numpy as np

def detect_rectangles(image_path, output_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Could not open or find the image.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection
    edged = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Get the minimum area rectangle for the contour
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)  # Changed to np.int32

        # Filter rectangles by area and aspect ratio if needed
        width, height = rect[1]
        angle = rect[2]
        if width > 100 and height > 100:
            aspect_ratio = width / height if width > height else height / width
            angle = abs(angle) if width > height else abs(angle + 90)
            if 75 <= angle <= 105 and 1.5 <= aspect_ratio <= 3:  # Adjust as necessary
                cv2.drawContours(image, [box], 0, (255, 255, 0), 15)  # Ensure color and thickness are appropriate

    # Save the image with rectangles
    cv2.imwrite(output_path, image)
    print(f"Output saved to {output_path}")

# Path to your image file
image_path = 'IMG_0641.jpg'
# Path to save the output image
output_path = 'result.jpg'

# Detect and save rectangular blobs
detect_rectangles(image_path, output_path)
