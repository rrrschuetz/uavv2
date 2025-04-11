import numpy as np
import torch
import cloudpickle

def apply_reciprocal(data):
    """Applies reciprocal transformation to the data, handling zeros."""
    with np.errstate(divide='ignore'):
        return np.where(data != 0.0, 1.0 / data, 0.0)

def preprocess_input(lidar_raw, color_raw, device):
    try:
        lidar_raw = lidar_raw.reshape(1, -1)
        # Apply reciprocal transformation to LIDAR data
        lidar_data = apply_reciprocal(lidar_raw)

        # No scaling is applied to lidar_data, use it as-is (no StandardScaler)
        lidar_data = lidar_data.astype(np.float32)  # Ensure it's float32

        # Reshape data for model input
        lidar_data = lidar_data.reshape(1, 1, lidar_data.shape[1])

        color_data = color_raw.astype(np.float32).reshape(1, 1, color_raw.shape[0])

        # Convert to PyTorch tensors
        lidar_tensor = torch.tensor(lidar_data).to(device)
        color_tensor = torch.tensor(color_data).to(device)

    except Exception as e:
        print(f"Error: {e}")
        lidar_tensor = None
        color_tensor = None

    return lidar_tensor, color_tensor
