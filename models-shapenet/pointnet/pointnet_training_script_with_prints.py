import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from tqdm import tqdm
import trimesh
import torch.nn.functional as F

# Function to sample or pad the point cloud to a fixed number of points
def fix_point_cloud_size(point_cloud, num_points=1024):
    if point_cloud.shape[0] > num_points:
        indices = np.random.choice(point_cloud.shape[0], num_points, replace=False)
        point_cloud = point_cloud[indices, :]
    elif point_cloud.shape[0] < num_points:
        indices = np.random.choice(point_cloud.shape[0], num_points - point_cloud.shape[0], replace=True)
        point_cloud = np.concatenate([point_cloud, point_cloud[indices, :]], axis=0)
    return point_cloud

# Dataset class for HermanMiller dataset
class HermanMillerDataset(Dataset):
    def __init__(self, labels, obj_folder, num_points=1024):
        self.labels = labels
        self.obj_folder = obj_folder
        self.num_points = num_points

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        first_layer_folder = str(int(row['Folder Name'])).zfill(8)
        second_layer_folder = str(row['Object ID (Dataset Original Object ID)']).strip()
        obj_filename = 'model_normalized'
        obj_file = os.path.join(self.obj_folder, first_layer_folder, second_layer_folder, 'models', f"{obj_filename}.obj")
        
        try:
            mesh = trimesh.load(obj_file)
            point_cloud = np.array(mesh.vertices)
        except Exception as e:
            print(f"Error loading {obj_file}: {e}")
            point_cloud = np.zeros((self.num_points, 3))

        point_cloud = fix_point_cloud_size(point_cloud, self.num_points)
        label = row['Final Regularity Level'] - 1
        return torch.tensor(point_cloud, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# PointNet++ layers for Set Abstraction and Feature Propagation
class PointNetSetAbstraction(nn.Module):
    def __init__(self, num_points, in_channels, out_channels):
        super(PointNetSetAbstraction, self).__init__()
        self.num_points = num_points
        self.conv1 = nn.Conv1d(in_channels, out_channels // 2, 1)
        self.conv2 = nn.Conv1d(out_channels // 2, out_channels, 1)
        self.bn1 = nn.BatchNorm1d(out_channels // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # Sampling and grouping points
        x = x[:, :self.num_points, :].transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PointNetFeaturePropagation, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        return x

class PointNetPlusPlus(nn.Module):
    def __init__(self, num_classes):
        super(PointNetPlusPlus, self).__init__()
        self.sa1 = PointNetSetAbstraction(num_points=512, in_channels=3, out_channels=128)
        self.sa2 = PointNetSetAbstraction(num_points=128, in_channels=128, out_channels=256)
        self.fp1 = PointNetFeaturePropagation(in_channels=256, out_channels=128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.sa1(x)
        x = self.sa2(x)
        x = self.fp1(x)
        x = x.max(dim=2)[0]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Training PointNet++ model
def train_pointnet_plus_plus(dataset_folder, excel_path, num_epochs=20, batch_size=32, num_points=1024, learning_rate=0.001):
    # Load labels
    label_data = pd.read_excel(excel_path)
    labels = label_data[['Object ID (Dataset Original Object ID)', 'Final Regularity Level', 'Folder Name']]

    # Create dataset and dataloaders
    dataset = HermanMillerDataset(labels, dataset_folder, num_points=num_points)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize PointNet++ model
    num_classes = len(labels['Final Regularity Level'].unique())
    model = PointNetPlusPlus(num_classes).to(config["device"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

    # Evaluation loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

# Path setup
dataset_folder = 'datasets/HermanMiller/obj-hermanmiller'
excel_path = 'datasets/HermanMiller/label/Final_Validated_Regularity_Levels.xlsx'

# Run the training
train_pointnet_plus_plus(dataset_folder, excel_path)
