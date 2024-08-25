import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load data without headers
data_raw = pd.read_csv("./data_file.txt", header=None)

# Extract lidar values (adjust column indices as needed)
lidar_data = data_raw.iloc[:, 2:1502].values  # Assuming these columns contain LIDAR data

# Number of samples to advance with each keystroke
step_size = 1


def plot_radar_chart(ax, lidar_values):
    """Plots a scatter plot on a radar chart for the LIDAR data."""
    # Create an array of angles equally spaced around the circle
    num_bins = len(lidar_values)
    angles = np.linspace(0, np.pi, num_bins, endpoint=False).tolist()

    # Clear previous plot
    ax.clear()

    # Plot data as scatter points
    ax.scatter(angles, lidar_values, s=10)  # s=10 sets the size of the scatter points

    # Set the labels and title
    ax.set_title("LIDAR Data Radar Chart")
    ax.set_yticklabels([])  # Hide radial labels
    ax.grid(True)

def update_plot(index):
    """Update plot with LIDAR data at a specific index."""
    if index >= len(lidar_data):
        print("End of data reached.")
        return False  # To stop animation
    plot_radar_chart(ax, lidar_data[index])
    fig.canvas.draw()
    return True

def on_key(event):
    """Handle key press events to advance the radar chart."""
    global current_index
    if event.key == 'n':  # 'n' key to move forward
        current_index += step_size
        if current_index >= len(lidar_data):
            current_index = len(lidar_data) - 1
        update_plot(current_index)
    elif event.key == 'q':  # 'q' key to quit
        plt.close()

# Initialize plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
current_index = 0

# Plot initial data
update_plot(current_index)

# Connect the key press event
fig.canvas.mpl_connect('key_press_event', on_key)

# Show plot
plt.show()
