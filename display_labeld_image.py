import cv2
import time
import matplotlib.pyplot as plt

def display_image_with_matplotlib(image, ax):
    # Convert the image from BGR (OpenCV format) to RGB (Matplotlib format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Update the image display
    ax.clear()
    ax.imshow(image_rgb)
    ax.axis('off')  # Hide the axis

def display_images(image_path, delay=1):
    # Create a figure and axis for the image
    fig, ax_image = plt.subplots(figsize=(6, 6))

    while True:
        # Read the image
        image = cv2.imread(image_path)
        print(f"Image loaded: {image is not None}")  # Debug print

        if image is None:
            print(f"Failed to load image: {image_path}")
        else:
            # Display the image using matplotlib
            display_image_with_matplotlib(image, ax_image)

        # Refresh the plot
        plt.pause(0.001)

        # Add a sleep time to control the reading interval
        time.sleep(delay / 1000.0)

# Example usage
display_images('/mnt/uavv2/labeled_image.jpg', 1000)
