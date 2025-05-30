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
    blob_min_width = width // 40
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[height//4:, :] = 255

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
    # Initialize the cameras
    picam0 = Picamera2(camera_num=0)
    picam1 = Picamera2(camera_num=1)

    # Configure the cameras with a higher resolution
    config = {"format": 'RGB888', "size": (1920, 1080)}
    picam0.configure(picam0.create_preview_configuration(main=config))
    picam1.configure(picam1.create_preview_configuration(main=config))

    # Start the cameras
    picam0.start()
    picam1.start()

    while True:
        # Capture images from both cameras
        image0 = picam0.capture_array()
        image1 = picam1.capture_array()
        image0_flipped = cv2.flip(image0, -1)
        image1_flipped = cv2.flip(image1, -1)

        # Combine the images side by side
        combined_image = np.hstack((image1_flipped, image0_flipped))

        # Detect and label rectangles on the combined image
        labeled_image, dilated_image = detect_and_label_rectangles(combined_image)

        # Resize the images to a scale of 1:4 for display
        s_labeled_image = cv2.resize(labeled_image, (0, 0), fx=0.25, fy=0.25)
        s_dilated_image = cv2.resize(dilated_image, (0, 0), fx=0.25, fy=0.25)

        # Display the images
        cv2.imshow('Combined Camera - Labeled', s_labeled_image)
        cv2.imshow('Combined Camera - Blob Detector', s_dilated_image)

        # Press 'q' on the keyboard to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the cameras
    picam2.stop()
    picam3.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
