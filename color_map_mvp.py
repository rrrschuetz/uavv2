from picamera2 import Picamera2, Preview
import cv2
import numpy as np
import time

def gamma_correction(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def enhance_lighting(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v = clahe.apply(v)
    hsv_enhanced = cv2.merge([h, s, v])
    enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
    return enhanced

def preprocess_image(image):
    gamma_corrected = gamma_correction(image)
    enhanced = enhance_lighting(gamma_corrected)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    return hsv

def apply_morphological_operations(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=4)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)
    return mask

def remove_small_contours(mask, min_area=2000):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            cv2.drawContours(mask, [contour], -1, 0, -1)
    return mask

def filter_contours(contours, min_area=2000, aspect_ratio_range=(1.5, 3.0), angle_range=(80, 100)):
    filtered_contours = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
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
    hsv = preprocess_image(image)

    # Adaptive color ranges for red and green detection
    red_lower1 = np.array([0, 50, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 50, 50])
    red_upper2 = np.array([180, 255, 255])
    green_lower1 = np.array([35, 40, 40])
    green_upper1 = np.array([70, 255, 255])
    green_lower2 = np.array([70, 40, 40])
    green_upper2 = np.array([90, 255, 255])

    # Create masks
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    green_mask1 = cv2.inRange(hsv, green_lower1, green_upper1)
    green_mask2 = cv2.inRange(hsv, green_lower2, green_upper2)
    green_mask = cv2.bitwise_or(green_mask1, green_mask2)

    # Apply morphological operations and remove small contours
    red_mask = remove_small_contours(apply_morphological_operations(red_mask))
    green_mask = remove_small_contours(apply_morphological_operations(green_mask))

    # Combine masks
    combined_mask = cv2.bitwise_or(red_mask, green_mask)

    # Find and filter contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = filter_contours(contours)

    im_with_keypoints = image.copy()
    blob_data = []

    for box in filtered_contours:
        cv2.drawContours(im_with_keypoints, [box], 0, (0, 255, 255), 5)
        rect = cv2.minAreaRect(box)
        center = (int(rect[0][0]), int(rect[0][1]))
        label = "R" if np.any(red_mask[center[1], center[0]]) else "G"
        label_position = (center[0], center[1])
        cv2.putText(im_with_keypoints, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        left_end = min(box[:, 0])
        right_end = max(box[:, 0])
        blob_data.append((left_end, right_end, label))

    return im_with_keypoints, red_mask, green_mask, blob_data

def main():
    picam0 = Picamera2(camera_num=0)
    picam1 = Picamera2(camera_num=1)
    config = {"format": 'RGB888', "size": (640, 400)}
    picam0.configure(picam0.create_preview_configuration(main=config))
    picam1.configure(picam1.create_preview_configuration(main=config))
    picam0.start()
    picam1.start()

    while True:
        start_time = time.time()
        image0 = picam0.capture_array()
        image1 = picam1.capture_array()
        image0_flipped = cv2.flip(image0, 0)
        image1_flipped = cv2.flip(image1, 0)
        combined_image = np.hstack((image1_flipped, image0_flipped))

        height = combined_image.shape[0]
        cropped_image = combined_image[height // 3:, :]

        labeled_image, red_mask, green_mask, blob_data = detect_and_label_blobs(cropped_image)
        cv2.imshow('Combined Camera - Blobs', labeled_image)
        cv2.imshow('Red Mask', red_mask)
        cv2.imshow('Green Mask', green_mask)

        for blob in blob_data:
            print(f"X-coordinates: Left end = {blob[0]}, Right end = {blob[1]}, Color = {blob[2]}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Ensure the loop runs at 10 fps
        elapsed_time = time.time() - start_time
        time_to_wait = max(0, (1 / 10) - elapsed_time)
        time.sleep(time_to_wait)

    picam0.stop()
    picam1.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
