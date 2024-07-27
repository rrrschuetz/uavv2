from picamera2 import Picamera2, Preview
import cv2
import numpy as np


def adjust_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    adjusted = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return adjusted


def preprocess_image(image):
    # Adjust contrast
    adjusted = adjust_contrast(image)

    # Convert to HSV color space for color filtering
    hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)

    return hsv


def apply_morphological_operations(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=4)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)
    return mask


def remove_small_contours(mask, min_area=1500):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            cv2.drawContours(mask, [contour], -1, 0, -1)
    return mask


def filter_contours(contours, min_area=1500, aspect_ratio_range=(1.5, 3.0), angle_range=(80, 100)):
    filtered_contours = []
    for contour in contours:
        # Filter by area
        if cv2.contourArea(contour) < min_area:
            continue

        # Get the minimum area rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        # Get width, height, and angle of the rectangle
        width, height = rect[1]
        angle = rect[2]
        if width < height:
            width, height = height, width
            angle += 90

        aspect_ratio = width / height
        if aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1] and angle_range[0] <= angle <= angle_range[1]:
            filtered_contours.append(box)

    return filtered_contours


def detect_and_label_blobs(image):
    # Preprocess the image
    hsv = preprocess_image(image)

    # Define the color ranges for red and green in HSV
    red_lower1 = np.array([0, 100, 100])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 100, 100])
    red_upper2 = np.array([180, 255, 255])
    green_lower = np.array([35, 50, 50])  # Further fine-tuned green criteria
    green_upper = np.array([85, 255, 255])

    # Create masks for red and green colors
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    # Apply morphological operations to reduce noise in the red and green masks
    red_mask = apply_morphological_operations(red_mask)
    green_mask = apply_morphological_operations(green_mask)

    # Remove small contours that are likely to be noise
    red_mask = remove_small_contours(red_mask)
    green_mask = remove_small_contours(green_mask)

    # Apply bilateral filter to preserve edges while reducing noise
    bilateral_filtered_image = cv2.bilateralFilter(image, 9, 75, 75)

    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection with adjusted parameters
    edges = cv2.Canny(gray, 50, 150)

    # Combine edge mask with color masks
    red_edges = cv2.bitwise_and(edges, red_mask)
    green_edges = cv2.bitwise_and(edges, green_mask)
    combined_edges = cv2.bitwise_or(red_edges, green_edges)

    # Combine edges with original masks
    red_mask = cv2.bitwise_or(red_mask, red_edges)
    green_mask = cv2.bitwise_or(green_mask, green_edges)

    # Combine red and green masks
    combined_mask = cv2.bitwise_or(red_mask, green_mask)

    # Find contours in the combined mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours for upright rectangular shapes
    filtered_contours = filter_contours(contours)

    # Draw yellow bounding boxes and labels above the detected blobs
    im_with_keypoints = image.copy()
    for box in filtered_contours:
        cv2.drawContours(im_with_keypoints, [box], 0, (0, 255, 255), 10)
        # Determine the label based on the color mask
        rect = cv2.minAreaRect(box)
        center = (int(rect[0][0]), int(rect[0][1]))
        label = "Red" if np.any(red_mask[center[1], center[0]]) else "Green"
        # Put the label above the bounding box
        label_position = (center[0] - 50, center[1] - 20)
        cv2.putText(im_with_keypoints, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 6)

    return im_with_keypoints, red_mask, green_mask


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

        # Flip the images vertically
        image0_flipped = cv2.flip(image0, 0)
        image1_flipped = cv2.flip(image1, 0)

        # Combine the images side by side
        combined_image = np.hstack((image1_flipped, image0_flipped))

        # Detect and label blobs on the combined image
        labeled_image, red_mask, green_mask = detect_and_label_blobs(combined_image)

        # Resize the images to a scale of 1:4 for display
        s_labeled_image = cv2.resize(labeled_image, (0, 0), fx=0.25, fy=0.25)
        s_red_mask = cv2.resize(red_mask, (0, 0), fx=0.25, fy=0.25)
        s_green_mask = cv2.resize(green_mask, (0, 0), fx=0.25, fy=0.25)

        # Display the images
        cv2.imshow('Combined Camera - Blobs', s_labeled_image)
        cv2.imshow('Red Mask', s_red_mask)
        cv2.imshow('Green Mask', s_green_mask)

        # Press 'q' on the keyboard to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the cameras
    picam0.stop()
    picam1.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
