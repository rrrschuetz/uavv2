import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle


def make_column_names_unique(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in
                                                         range(sum(cols == dup))]
    df.columns = cols


def apply_reciprocal_to_scan(df):
    scan_cols = df.filter(regex='^SCAN').columns
    for col in scan_cols:
        df[col] = df[col].apply(lambda x: 1 / x if x != 0 else 0)
    return df


def add_gaussian_noise(data, mean=0.0, stddev=1.0):
    noise = np.random.normal(mean, stddev, size=data.shape)
    noisy_data = data + noise
    return noisy_data


def apply_dropout(data, dropout_rate=0.1):
    mask = np.random.binomial(1, 1 - dropout_rate, size=data.shape)
    data_with_dropout = data * mask
    return data_with_dropout


# 1. Preprocess data
data_raw = pd.read_csv("~/test/file.txt")
make_column_names_unique(data_raw)
data_raw = apply_reciprocal_to_scan(data_raw)
lidar_cols = data_raw.filter(regex='^SCAN').columns
noisy_data = data_raw.copy()
for col in lidar_cols:
    noisy_data[col] = add_gaussian_noise(data_raw[col], mean=0.0, stddev=0.01)
data_raw = pd.concat([data_raw, noisy_data], axis=0).reset_index(drop=True)
print("Raw data columns:", data_raw.columns)
print("Raw data shape:", data_raw.shape)

# Split data into train and test sets
train, test = train_test_split(data_raw, test_size=0.2)
print("Train data shape:", train.shape)
print("Test data shape:", test.shape)

train_lidar = train.iloc[:, 2:1622]
test_lidar = test.iloc[:, 2:1622]
train_color = train.iloc[:, -1280:]
test_color = test.iloc[:, -1280:]
y_train = train.iloc[:, 0:2].values
y_test = test.iloc[:, 0:2].values

# Standardization
scaler_lidar = StandardScaler().fit(train_lidar.values)
print("Scaler fitted on x_train")
train_lidar = scaler_lidar.transform(train_lidar.values).astype(np.float32)
test_lidar = scaler_lidar.transform(test_lidar.values).astype(np.float32)

# Convert to numpy arrays and reshape as needed for LIDAR data
train_lidar = train_lidar.reshape(train_lidar.shape[0], 1, train_lidar.shape[1])
test_lidar = test_lidar.reshape(test_lidar.shape[0], 1, test_lidar.shape[1])
train_color = train_color.values.astype(np.float32).reshape(train_color.shape[0], 1, train_color.shape[1])
test_color = test_color.values.astype(np.float32).reshape(test_color.shape[0], 1, test_color.shape[1])

print("After standardization, train lidar shape:", train_lidar.shape)
print("After standardization, test lidar shape:", test_lidar.shape)
print("After standardization, train color shape:", train_color.shape)
print("After standardization, test color shape:", test_color.shape)

# Convert data to PyTorch tensors
train_lidar = torch.tensor(train_lidar)
test_lidar = torch.tensor(test_lidar)
train_color = torch.tensor(train_color)
test_color = torch.tensor(test_color)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# 2. Define the 1D CNN model
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
        self.lidar_conv1 = nn.Conv1d(1, 64, kernel_size=5, activation='relu')
        self.lidar_pool1 = nn.MaxPool1d(2)
        self.lidar_conv2 = nn.Conv1d(64, 128, kernel_size=5, activation='relu')
        self.lidar_pool2 = nn.MaxPool1d(2)
        self.lidar_flatten = nn.Flatten()

        self.color_dense1 = nn.Linear(color_input_shape[0], 64)
        self.color_dropout = nn.Dropout(0.3)
        self.color_dense2 = nn.Linear(64, 128)
        self.color_flatten = nn.Flatten()

        self.weighted_concat = WeightedConcatenate(weight_lidar=0.1, weight_color=0.9)

        self.fc1 = nn.Linear(128 + 128, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 2)

    def forward(self, lidar, color):
        lidar = self.lidar_conv1(lidar)
        lidar = self.lidar_pool1(lidar)
        lidar = self.lidar_conv2(lidar)
        lidar = self.lidar_pool2(lidar)
        lidar = self.lidar_flatten(lidar)

        color = self.color_dense1(color)
        color = self.color_dropout(color)
        color = self.color_dense2(color)
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
model = CNNModel(lidar_input_shape, color_input_shape)

# 3. Train the model
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
early_stopping_patience = 2
best_loss = float('inf')
epochs_no_improve = 0

for epoch in range(45):
    model.train()
    optimizer.zero_grad()
    output = model(train_lidar, train_color)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_output = model(test_lidar, test_color)
        val_loss = criterion(val_output, y_test)

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}, Val Loss: {val_loss.item()}")

    if val_loss < best_loss:
        best_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= early_stopping_patience:
            print("Early stopping")
            break

# 4. Evaluate the model
model.load_state_dict(torch.load('best_model.pt'))
model.eval()
with torch.no_grad():
    test_output = model(test_lidar, test_color)
    test_loss = criterion(test_output, y_test)
    accuracy = ((test_output.argmax(dim=1) == y_test.argmax(dim=1)).float().mean()).item()

print(f"Test Loss: {test_loss.item()}")
print(f"Test Accuracy: {accuracy}")

# 5. Save the model and the scaler for standardization
torch.save(model.state_dict(), '/home/rrrschuetz/test/model.pth')
with open('/home/rrrschuetz/test/scaler.pkl', 'wb') as f:
    pickle.dump(scaler_lidar, f)
