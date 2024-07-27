from picamera2 import Picamera2, Preview
import cv2
import numpy as np

def adjust_contrast(image):
    alpha = 1.5  # Simple contrast control
    beta = 0    # Simple brightness control
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def preprocess_image(image):
    # Adjust contrast
    adjusted = adjust_contrast(image)

    # Convert to grayscale
    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    return blurred

def detect_edges(blurred):
    # Perform edge detection with adjusted parameters
    edged = cv2.Canny(blurred, 50, 150)

    # Apply morphological transformations to close gaps and enhance edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)

    return closed

def filter_contours(contours, image_shape):
    height, width = image_shape[:2]
    blob_min_height = height // 20
    blob_min_width = width // 20
    filtered_contours = []

    for contour in contours:
        rect = cv2.minAreaRect(contour)
        rect_width, rect_height = rect[1]
        angle = rect[2]

        if rect_width > blob_min_width and rect_height > blob_min_height:
            aspect_ratio = rect_width / rect_height if rect_width > rect_height else rect_height / rect_width
            angle = abs(angle) if rect_width > rect_height else abs(angle + 90)
            if 80 <= angle <= 100 and 1.5 <= aspect_ratio <= 3:
                filtered_contours.append(contour)

    return filtered_contours

def detect_and_label_rectangles(image):
    # Ignore the upper part of the image
    height, width, _ = image.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[height//2:, :] = 255

    # Preprocess the image
    blurred = preprocess_image(image)

    # Detect edges
    closed = detect_edges(blurred)

    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours
    filtered_contours = filter_contours(contours, image.shape)

    for contour in filtered_contours:
        # Get the minimum area rectangle for the contour
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        # Analyze the color composition within the inner bounding box
        inner_rect = cv2.boundingRect(contour)
        x, y, w, h = inner_rect
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
        # Draw the inner bounding rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)
        # Put the label near the bounding rectangle
        cv2.putText(image, label, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0), 10)
        print(f"Detected - Width: {rect[1][0]}, Height: {rect[1][1]}, Inner Width: {w}, Inner Height: {h}, Angle: {angle}, Aspect Ratio: {aspect_ratio}, Label: {label}")

    return image, closed

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
        imageß = picam0.capture_array()
        image1 = picam1.capture_array()

        # Flip the images vertically
        image0_flipped = cv2.flip(imageß, -1)
        image1_flipped = cv2.flip(image1, -1)

        # Combine the images side by side
        combined_image = np.hstack((image1_flipped, image0_flipped))

        # Detect and label rectangles on the combined image
        labeled_image, closed_image = detect_and_label_rectangles(combined_image)

        # Resize the images to a scale of 1:4 for display
        s_labeled_image = cv2.resize(labeled_image, (0, 0), fx=0.25, fy=0.25)
        s_closed_image = cv2.resize(closed_image, (0, 0), fx=0.25, fy=0.25)

        # Display the images
        cv2.imshow('Combined Camera - Labeled', s_labeled_image)
        cv2.imshow('Combined Camera - Edge Detection', s_closed_image)

        # Press 'q' on the keyboard to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the cameras
    picam0.stop()
    picam1.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
