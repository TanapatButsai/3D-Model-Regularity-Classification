import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from tqdm import tqdm
import trimesh
import random

# Configuration dictionary
config = {
    "num_samples": 7500,
    "num_epochs": 50,
    "batch_size": 32,
    "num_points": 1024,
    "learning_rate": 0.001,
    "dataset_folder": 'datasets/ShapeNetCoreV2/obj-ShapeNetCoreV2',
    "excel_path": 'datasets/ShapeNetCoreV2/label/final_regularized_labels.xlsx',
    "patience": 5,  # For early stopping
}

# Function to sample or pad the point cloud to a fixed number of points
def fix_point_cloud_size(point_cloud, num_points=1024):
    if point_cloud.shape[0] > num_points:
        indices = np.random.choice(point_cloud.shape[0], num_points, replace=False)
        point_cloud = point_cloud[indices, :]
    elif point_cloud.shape[0] < num_points:
        indices = np.random.choice(point_cloud.shape[0], num_points - point_cloud.shape[0], replace=True)
        point_cloud = np.concatenate([point_cloud, point_cloud[indices, :]], axis=0)
    return point_cloud

# Function to augment point cloud data
def augment_point_cloud(point_cloud):
    # Random rotation around the z-axis
    theta = random.uniform(0, 2 * np.pi)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    point_cloud = np.dot(point_cloud, rotation_matrix)
    # Random scaling
    scale = random.uniform(0.9, 1.1)
    point_cloud *= scale
    return point_cloud

# Function to extract features from OBJ files
def extract_features_from_obj(file_path, num_points=1024):
    try:
        loaded_obj = trimesh.load(file_path)
        if isinstance(loaded_obj, trimesh.Scene):
            if loaded_obj.geometry:
                mesh = trimesh.util.concatenate(loaded_obj.geometry.values())
            else:
                print(f"Scene has no geometries: {file_path}")
                return None
        else:
            mesh = loaded_obj
        if not hasattr(mesh, 'vertices'):
            print(f"Mesh has no vertices: {file_path}")
            return None
        point_cloud = np.array(mesh.vertices)
        point_cloud = fix_point_cloud_size(point_cloud, num_points)
        point_cloud = augment_point_cloud(point_cloud)  # Apply data augmentation
        return point_cloud
    except ValueError as ve:
        print(f"ValueError loading {file_path}: {ve}")
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

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
        point_cloud = extract_features_from_obj(obj_file, self.num_points)
        if point_cloud is None:
            point_cloud = np.zeros((self.num_points, 3))
        label = row['Final Regularity Level'] - 1
        return torch.tensor(point_cloud, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Enhanced PointNet model for increased capacity
class EnhancedPointNet(nn.Module):
    def __init__(self, num_classes):
        super(EnhancedPointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 128, 1)  # Increased layer sizes
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.conv3 = nn.Conv1d(256, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.maxpool = nn.MaxPool1d(1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # Adding dropout for regularization

    def forward(self, x):
        x = x.transpose(2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        x = x.squeeze(-1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# Function to train the Enhanced PointNet model with early stopping and learning rate scheduler
def train_pointnet(config):
    label_data = pd.read_excel(config["excel_path"])
    labels = label_data[['Object ID (Dataset Original Object ID)', 'Final Regularity Level', 'Folder Name']]
    if config["num_samples"] < len(labels):
        labels = labels.sample(n=config["num_samples"], random_state=42)
    
    dataset = ShapeNetDataset(labels, config["dataset_folder"], num_points=config["num_points"])
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    num_classes = len(labels['Final Regularity Level'].unique())
    model = EnhancedPointNet(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_loss = float('inf')
    patience = 0

    for epoch in range(config["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{config['num_epochs']}], Loss: {avg_epoch_loss:.4f}")

        # Early stopping check
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience = 0
        else:
            patience += 1
            if patience >= config["patience"]:
                print("Early stopping triggered.")
                break

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

# Run the training with configuration
train_pointnet(config)
