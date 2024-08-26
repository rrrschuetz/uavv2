
import torch
import torch.nn as nn

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

        conv_output_size = self.calculate_conv_output_size(lidar_input_shape)

        self.lidar_flatten = nn.Flatten()

        self.color_dense1 = nn.Linear(color_input_shape, 64)
        self.color_dropout = nn.Dropout(0.3)
        self.color_dense2 = nn.Linear(64, 128)
        self.color_flatten = nn.Flatten()

        self.weighted_concat = WeightedConcatenate(weight_lidar=0.1, weight_color=0.9)
        self.fc1 = nn.Linear(conv_output_size + 128, 64)

        self.fc1 = nn.Linear(conv_output_size, 64)
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
        #combined = torch.relu(self.fc1(lidar))
        combined = torch.relu(self.fc2(combined))
        combined = torch.relu(self.fc3(combined))
        combined = torch.relu(self.fc4(combined))
        combined = torch.relu(self.fc5(combined))
        output = self.output(combined)
        return output
