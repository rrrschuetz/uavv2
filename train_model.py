import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle

def apply_reciprocal(data):
    """Applies reciprocal transformation to the data, handling zeros."""
    with np.errstate(divide='ignore'):
        return np.where(data != 0.0, 1.0 / data, 0.0)

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

# Define the 1D CNN model
class WeightedConcatenate(nn.Module):
    def __init__(self, weight_lidar=0.5, weight_color=0.5):
        super(WeightedConcatenate, self).__init__()
        self.weight_lidar = weight_lidar
        self.weight_color = weight_color

    def forward(self, lidar, color):
        return torch.cat([self.weight_lidar * lidar, self.weight_color * color], dim=-1)

class CNNModel(nn.Module):
    def __init__(self, lidar_input_shape, color_input_shape):
        super(CNNModel, self).__init__()
        self.lidar_conv1 = nn.Conv1d(1, 64, kernel_size=5)
        self.lidar_pool1 = nn.MaxPool1d(2)
        self.lidar_conv2 = nn.Conv1d(64, 128, kernel_size=5)
        self.lidar_pool2 = nn.MaxPool1d(2)

        # Calculate the output size after convolutions and pooling
        conv_output_size = self.calculate_conv_output_size(lidar_input_shape)

        self.lidar_flatten = nn.Flatten()

        self.color_dense1 = nn.Linear(color_input_shape, 64)
        self.color_dropout = nn.Dropout(0.3)
        self.color_dense2 = nn.Linear(64, 128)
        self.color_flatten = nn.Flatten()

        self.weighted_concat = WeightedConcatenate(weight_lidar=0.1, weight_color=0.9)

        # Update the input size of the first fully connected layer
        self.fc1 = nn.Linear(conv_output_size + 128, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 2)

    def calculate_conv_output_size(self, input_size):
        size = (input_size - 4) // 2  # After first conv and pool
        size = (size - 4) // 2  # After second conv and pool
        return size * 128

    def forward(self, lidar, color):
        lidar = torch.relu(self.lidar_conv1(lidar))
        lidar = self.lidar_pool1(lidar)
        lidar = torch.relu(self.lidar_conv2(lidar))
        lidar = self.lidar_pool2(lidar)
        lidar = self.lidar_flatten(lidar)

        color = torch.relu(self.color_dense1(color))
        color = self.color_dropout(color)
        color = torch.relu(self.color_dense2(color))
        color = self.color_flatten(color)

        concatenated = self.weighted_concat(lidar, color)

        combined = torch.relu(self.fc1(concatenated))
        combined = torch.relu(self.fc2(combined))
        combined = torch.relu(self.fc3(combined))
        combined = torch.relu(self.fc4(combined))
        combined = torch.relu(self.fc5(combined))
        output = self.output(combined)
        return output

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

# Save the model and the scaler for standardization
torch.save(model.state_dict(), './model.pth')
with open('./scaler.pkl', 'wb') as f:
    pickle.dump(scaler_lidar, f)
