import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from tqdm import tqdm
import trimesh

# Function to sample or pad the point cloud to a fixed number of points
def fix_point_cloud_size(point_cloud, num_points=1024):
    if point_cloud.shape[0] > num_points:
        # Downsample to the desired number of points
        indices = np.random.choice(point_cloud.shape[0], num_points, replace=False)
        point_cloud = point_cloud[indices, :]
    elif point_cloud.shape[0] < num_points:
        # Upsample by repeating points
        indices = np.random.choice(point_cloud.shape[0], num_points - point_cloud.shape[0], replace=True)
        point_cloud = np.concatenate([point_cloud, point_cloud[indices, :]], axis=0)
    
    return point_cloud

# Dataset class for ShapeNet
class ShapeNetDataset(Dataset):
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
            point_cloud = np.zeros((self.num_points, 3))  # Default to zeros if there's an error
        
        # Ensure point cloud has exactly num_points
        point_cloud = fix_point_cloud_size(point_cloud, self.num_points)
        
        label = row['Final Regularity Level'] - 1  # Adjust label to start from 0
        return torch.tensor(point_cloud, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Simple PointNet implementation for demonstration
class PointNet(nn.Module):
    def __init__(self, num_classes):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.maxpool = nn.MaxPool1d(1024)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(2, 1)  # Change shape to (batch_size, 3, num_points)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        x = x.squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to train the PointNet model
def train_pointnet(dataset_folder, excel_path, num_epochs=20, batch_size=32, num_points=1024, learning_rate=0.001):
    # Load labels from Excel file
    label_data = pd.read_excel(excel_path)
    labels = label_data[['Object ID (Dataset Original Object ID)', 'Final Regularity Level', 'Folder Name']]
    
    # Create dataset and dataloaders
    dataset = ShapeNetDataset(labels, dataset_folder, num_points=num_points)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Total dataset size: {len(dataset)} samples")
    print(f"Training set size: {train_size} samples")
    print(f"Test set size: {test_size} samples")
    print(f"Batch size: {batch_size}")
    
    # Initialize PointNet model
    num_classes = len(labels['Final Regularity Level'].unique())
    model = PointNet(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, (inputs, labels) in enumerate(tqdm(train_loader)):
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
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

# Path setup
dataset_folder = 'datasets/ShapeNetCoreV2/obj-ShapeNetCoreV2'
excel_path = 'datasets/ShapeNetCoreV2/label/Final_Validated_Regularity_Levels.xlsx'

# Run the training
train_pointnet(dataset_folder, excel_path)
