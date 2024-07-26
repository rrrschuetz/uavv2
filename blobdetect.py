import cv2
import numpy as np

def detect_rectangles(image_path, output_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Could not open or find the image.")
        return

    alpha = 1.5 # Simple contrast control
    beta = 0  # Simple brightness control
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    cv2.imwrite(contrast_output_path, adjusted)

    # Convert to grayscale
    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(blurred, 50, 200)  # Adjusted the Canny edge detection thresholds
    cv2.imwrite(edge_output_path, edged)  # Save the edged image for inspection
    print(f"Edged image saved to {edge_output_path}")

    # Apply morphological operations to strengthen edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Adjusted kernel size
    dilated = cv2.dilate(edged, kernel, iterations=2)  # Increased iterations
    cv2.imwrite(dilated_output_path, dilated)  # Save the dilated image for inspection
    print(f"Dilated image saved to {dilated_output_path}")

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
# Path to save the edged image
edge_output_path = 'edged_image.jpg'
# Path to save the dilated image
dilated_output_path = 'dilated_image.jpg'
# Path to save the contrast adjusted image
contrast_output_path = 'contrast_adjusted_image.jpg'

# Detect and save rectangular blobs
detect_rectangles(image_path, output_path)
