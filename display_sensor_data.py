import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

def display_image_with_matplotlib(image, ax):
    # Convert the image from BGR (OpenCV format) to RGB (Matplotlib format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Update the image display
    ax.clear()
    ax.imshow(image_rgb)
    ax.axis('off')  # Hide the axis

def load_arrays_from_file(filename):
    data = np.loadtxt(filename, skiprows=1)
    distances = data[:, 0]
    angles = data[:, 1]
    return distances, angles

def draw_radar_chart(ax, distances, angles):
    ax.clear()
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2.0)
    ax.scatter(angles, distances, s=2)
    #ax.plot(angles, distances, linestyle='-', marker='o', markersize=2)
    ax.set_ylim(0, max(distances) * 1.1)
    ax.set_xlabel('Angle (radians)')
    ax.set_ylabel('Distance (meters)')
    ax.set_title('LIDAR Data')

def display_images(image_path, radar_path, delay=1):
    # Create a figure and two axes: one for the image, one for the radar chart
    fig = plt.figure(figsize=(12, 6))
    ax_image = fig.add_subplot(1, 2, 1)  # Regular axis for image
    ax_radar = fig.add_subplot(1, 2, 2, projection='polar')  # Polar axis for radar

    while True:
        # Read the image
        image = cv2.imread(image_path)
        print(f"Image loaded: {image is not None}")  # Debug print

        if image is None:
            print(f"Failed to load image: {image_path}")
        else:
            # Display the image using matplotlib
            display_image_with_matplotlib(image, ax_image)

        # Read and plot the radar data
        try:
            distances, angles = load_arrays_from_file(radar_path)
            distances = np.array(distances)
            angles = np.radians(angles)
            # Display the radar chart using matplotlib
            draw_radar_chart(ax_radar, distances, angles)

        except Exception as e:
            print(f"Failed to load radar data: {radar_path} {e}")

        # Refresh the plot
        plt.pause(0.001)

        # Add a sleep time to control the reading interval
        time.sleep(delay / 1000.0)

# Example usage
display_images('/mnt/uavv2/labeled_image.jpg', '/mnt/uavv2/radar.txt', 1000)

