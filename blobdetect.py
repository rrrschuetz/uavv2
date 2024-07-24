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

    # List to store rectangular blobs
    rectangles = []

    for contour in contours:
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # If the approximated contour has 4 vertices and is large enough, it is considered a rectangle
        if len(approx) == 4 and cv2.contourArea(approx) > 100:
            # Further filter by aspect ratio if needed
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 0.1 <= aspect_ratio <= 1.5:  # Adjust as necessary
                rectangles.append(approx)

    # Draw rectangles on the original image
    for rect in rectangles:
        cv2.drawContours(image, [rect], 0, (0, 255, 0), 10)

    # Save the image with rectangles
    cv2.imwrite(output_path, image)
    print(f"Output saved to {output_path}")

# Path to your image file
image_path = 'IMG_0641.jpg'
# Path to save the output image
output_path = 'result.jpg'

# Detect and save rectangular blobs
detect_rectangles(image_path, output_path)
