import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

def load_arrays_from_file(filename):
    data = np.loadtxt(filename, skiprows=1)
    distances = data[:, 0]
    angles = data[:, 1]
    return distances, angles

def draw_radar_chart(distances, angles):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2.0)

    ax.scatter(angles, distances, s=2)
    ax.set_ylim(0, max(distances) * 1.1)

    ax.set_xlabel('Angle (radians)')
    ax.set_ylabel('Distance (meters)')
    ax.set_title('LIDAR Data')

    plt.show()

def display_images(image_path, radar_path, delay=1):
    while True:
        # Read the images
        #image = cv2.imread(image_path)
        image = None

        if image is None:
            print(f"Failed to load image: {image_path}")
        else:
            # Display the first image in a window
            cv2.imshow('Image', image)

        # Read the radar data
        distances, angles = load_arrays_from_file(radar_path)
        print('Distances:', distances)
        draw_radar_chart(distances, angles)

        # Wait for the specified delay time in milliseconds
        key = cv2.waitKey(delay)
        if key == ord('q'):
            break

        # Add a sleep time to control the reading interval
        time.sleep(delay / 1000.0)

    # Close all OpenCV windows
    cv2.destroyAllWindows()

display_images('/mnt/uavv2/labeled_image.jpg', '/mnt/uavv2/radar.txt')
