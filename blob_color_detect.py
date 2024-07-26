import cv2
import numpy as np

def adjust_contrast(image):
    alpha = 1.5  # Simple contrast control
    beta = 0    # Simple brightness control
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def detect_and_label_rectangles(image_path, output_path, edge_output_path, contrast_output_path, dilated_output_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Could not open or find the image.")
        return

    # Ignore the upper third of the image
    height, width, _ = image.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[height//4:, :] = 255

    # Adjust contrast
    adjusted = adjust_contrast(image)
    cv2.imwrite(contrast_output_path, adjusted)
    print(f"Contrast adjusted image saved to {contrast_output_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=mask)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection with adjusted parameters
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
        box = np.int32(box)

        # Filter rectangles by area, aspect ratio, and angle
        width, height = rect[1]
        angle = rect[2]
        if width > 100 and height > 100:
            aspect_ratio = width / height if width > height else height / width
            angle = abs(angle) if width > height else abs(angle + 90)
            if 75 <= angle <= 105 and 1.5 <= aspect_ratio <= 3:
                # Analyze the color composition within the bounding box
                x, y, w, h = cv2.boundingRect(contour)
                roi = image[y:y+h, x:x+w]
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                # Define the color ranges for red and green in HSV
                red_lower1 = np.array([0, 50, 50])
                red_upper1 = np.array([10, 255, 255])
                red_lower2 = np.array([170, 50, 50])
                red_upper2 = np.array([180, 255, 255])
                green_lower = np.array([35, 50, 50])
                green_upper = np.array([85, 255, 255])

                red_mask1 = cv2.inRange(hsv_roi, red_lower1, red_upper1)
                red_mask2 = cv2.inRange(hsv_roi, red_lower2, red_upper2)
                red_mask = cv2.bitwise_or(red_mask1, red_mask2)
                green_mask = cv2.inRange(hsv_roi, green_lower, green_upper)

                red_count = cv2.countNonZero(red_mask)
                green_count = cv2.countNonZero(green_mask)

                label = "Red" if red_count > green_count else "Green"

                # Draw the bounding rectangle
                cv2.drawContours(image, [box], 0, (255, 0, 0), 10)
                # Put the label near the bounding rectangle
                cv2.putText(image, label, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0), 10)
                print(f"Detected - Width: {width}, Height: {height}, Angle: {angle}, Aspect Ratio: {aspect_ratio}, Label: {label}")

    # Save the image with rectangles and labels
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

# Detect and save vertically oriented rectangular blobs
detect_and_label_rectangles(image_path, output_path, edge_output_path, contrast_output_path, dilated_output_path)
