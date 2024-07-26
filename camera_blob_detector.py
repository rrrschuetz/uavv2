from picamera2 import Picamera2, Preview
import cv2
import numpy as np

def adjust_contrast(image):
    alpha = 1.5  # Simple contrast control
    beta = 0    # Simple brightness control
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def detect_and_label_rectangles(image):
    # Ignore the upper part of the image
    height, width, _ = image.shape
    blob_min_height = height // 20
    blob_min_width = width // 20
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[height//2:, :] = 255

    # Adjust contrast
    adjusted = adjust_contrast(image)

    # Convert to grayscale
    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=mask)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection with adjusted parameters
    edged = cv2.Canny(blurred, 50, 150)

    # Apply morphological operations to strengthen edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #eroded = cv2.erode(edged, kernel, iterations=1)  # Erode to thin the lines
    dilated = cv2.dilate(edged, kernel, iterations=1)  # Dilate to recover shapes

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
        if width > blob_min_width and height > blob_min_height:
            aspect_ratio = width / height if width > height else height / width
            angle = abs(angle) if width > height else abs(angle + 90)
            if 80 <= angle <= 100 and 1.5 <= aspect_ratio <= 3:
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

    return image, dilated

def main():
    # Initialize the camera
    picam2 = Picamera2()

    # Configure the camera
    #picam2.configure(picam2.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)}))
    picam2.configure(picam2.create_preview_configuration(main={"format": 'RGB888', "size": (1920, 1080)}))

    # Start the camera
    picam2.start()

    while True:
        # Capture an image
        image = picam2.capture_array()

        # Detect and label rectangles
        labeled_image, dilated_image = detect_and_label_rectangles(image)

        s_labeled_image = cv2.resize(labeled_image, (0, 0), fx=0.25, fy=0.25)
        s_dilated_image = cv2.resize(dilated_image, (0, 0), fx=0.25, fy=0.25)

        # Display the image
        cv2.imshow('Camera', s_labeled_image)
        cv2.imshow('Blob Detector', s_dilated_image)

        # Press 'q' on the keyboard to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera
    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
