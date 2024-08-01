import cv2
import time

def display_images(image_path1, image_path2, delay=1):
    while True:
        # Read the images
        image1 = cv2.imread(image_path1)
        image2 = cv2.imread(image_path2)

        if image1 is None:
            print(f"Failed to load image: {image_path1}")
        else:
            # Display the first image in a window
            cv2.imshow('Image 1', image1)

        if image2 is None:
            print(f"Failed to load image: {image_path2}")
        else:
            # Display the second image in a separate window
            cv2.imshow('Image 2', image2)

        # Wait for the specified delay time in milliseconds
        key = cv2.waitKey(delay)
        if key == ord('q'):
            break

        # Add a sleep time to control the reading interval
        time.sleep(delay / 1000.0)

    # Close all OpenCV windows
    cv2.destroyAllWindows()


# Example usage
image_path1 = '/mnt/uavv2/radar_chart.jpg'
image_path2 = '/mnt/uavv2/labeled_image.jpg'
display_images(image_path1, image_path2)
