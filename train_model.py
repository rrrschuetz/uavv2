import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from lidar_color_model import CNNModel  # Import the model from model.py
from preprocessing import apply_reciprocal  # Import preprocessing function
from sklearn.preprocessing import StandardScaler
import pickle

# Load data without headers
data_raw = pd.read_csv("./data_file.txt", header=None)

# Extract x, y, lidar, red, and green values
x_y = data_raw.iloc[:, :2].values
lidar_data = data_raw.iloc[:, 2:1502].values  # Adjust this based on your actual data range
red_values = data_raw.iloc[:, 1502:2782].values  # Adjust based on actual data range
green_values = data_raw.iloc[:, 2782:4062].values  # Adjust based on actual data range

# Apply reciprocal transformation to LIDAR data
lidar_data = apply_reciprocal(lidar_data)

# Standardize LIDAR data
scaler_lidar = StandardScaler().fit(lidar_data)
lidar_data = scaler_lidar.transform(lidar_data).astype(np.float32)

# Save the scaler for later use in inference
with open('./scaler.pkl', 'wb') as f:
    pickle.dump(scaler_lidar, f)

# Reshape data for model input
lidar_data = lidar_data.reshape(lidar_data.shape[0], 1, lidar_data.shape[1])
red_values = red_values.astype(np.float32).reshape(red_values.shape[0], 1, red_values.shape[1])
green_values = green_values.astype(np.float32).reshape(green_values.shape[0], 1, green_values.shape[1])

# Concatenate red and green values
color_data = np.concatenate((red_values, green_values), axis=2)

# Split data into train and test sets
train_lidar, test_lidar, train_color, test_color, y_train, y_test = train_test_split(
    lidar_data, color_data, x_y, test_size=0.2, random_state=42
)

# Convert data to PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_lidar = torch.tensor(train_lidar).to(device)
test_lidar = torch.tensor(test_lidar).to(device)
train_color = torch.tensor(train_color).to(device)
test_color = torch.tensor(test_color).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# Create DataLoader
batch_size = 128
train_dataset = TensorDataset(train_lidar, train_color, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(test_lidar, test_color, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Create the model
lidar_input_shape = train_lidar.shape[2]
color_input_shape = train_color.shape[2]
model = CNNModel(lidar_input_shape, color_input_shape).to(device)

# Train the model
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
early_stopping_patience = 2
best_loss = float('inf')
epochs_no_improve = 0

for epoch in range(45):
    model.train()
    for batch_lidar, batch_color, batch_labels in train_loader:
        optimizer.zero_grad()
        output = model(batch_lidar, batch_color)
        loss = criterion(output, batch_labels)
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_lidar, batch_color, batch_labels in test_loader:
            val_output = model(batch_lidar, batch_color)
            val_loss += criterion(val_output, batch_labels).item()

    val_loss /= len(test_loader)
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}, Val Loss: {val_loss}")

    if val_loss < best_loss:
        best_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= early_stopping_patience:
            print("Early stopping")
            break

# Evaluate the model
model.load_state_dict(torch.load('best_model.pt'))
model.eval()
test_loss = 0
accuracy = 0
with torch.no_grad():
    for batch_lidar, batch_color, batch_labels in test_loader:
        test_output = model(batch_lidar, batch_color)
        test_loss += criterion(test_output, batch_labels).item()
        accuracy += ((test_output.argmax(dim=1) == batch_labels.argmax(dim=1)).float().mean()).item()

test_loss /= len(test_loader)
accuracy /= len(test_loader)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {accuracy}")

# Save the final model
torch.save(model.state_dict(), './model.pth')