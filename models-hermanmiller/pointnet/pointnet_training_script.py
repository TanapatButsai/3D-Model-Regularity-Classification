
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import trimesh
import pandas as pd

# PointNet Model Definition
class PointNet(nn.Module):
    def __init__(self, k=4):
        super(PointNet, self).__init__()
        
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(1024)
        self.dropout = nn.Dropout(p=0.3)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        x = self.maxpool(x)
        x = x.view(-1, 1024)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# Convert OBJ file to point cloud
def obj_to_pointcloud(obj_path, num_points=1024):
    mesh = trimesh.load(obj_path)
    points = mesh.sample(num_points)
    return points

# Process the dataset to convert OBJ files into point clouds
def process_dataset(dataset_folder, excel_data, num_points=1024):
    point_clouds = []
    labels = []
    
    for index, row in excel_data.iterrows():
        object_id = row['Object ID (Dataset Original Object ID)']
        label = row['Final Regularity Level']
        
        obj_folder = os.path.join(dataset_folder, object_id)
        obj_file = os.path.join(obj_folder, f"{object_id}.obj")
        
        if os.path.exists(obj_file):
            point_cloud = obj_to_pointcloud(obj_file, num_points=num_points)
            point_clouds.append(point_cloud)
            labels.append(label)
    
    return np.array(point_clouds), np.array(labels)

# Main training loop
def train_pointnet(dataset_folder, excel_path, num_epochs=5):
    excel_data = pd.read_excel(excel_path)
    
    # Preprocess dataset
    point_clouds, labels = process_dataset(dataset_folder, excel_data)
    
    # Convert to tensors
    point_clouds_tensor = torch.tensor(point_clouds, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    # Define dataset
    class PointCloudDataset(torch.utils.data.Dataset):
        def __init__(self, point_clouds, labels):
            self.point_clouds = point_clouds
            self.labels = labels
        
        def __len__(self):
            return len(self.point_clouds)
        
        def __getitem__(self, idx):
            return self.point_clouds[idx], self.labels[idx]
    
    dataset = PointCloudDataset(point_clouds_tensor, labels_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Initialize model, optimizer, and loss function
    device = torch.device("cpu")
    model = PointNet(k=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("start")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, target in dataloader:
            inputs = inputs.transpose(2, 1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")
    print("Training completed!")

# Example usage
if __name__ == "__main__":
    dataset_folder = "datasets/hermanmiller/obj-hermanmiller"
    excel_path = "datasets/hermanmiller/label/Final_Validated_Regularity_Levels.xlsx"
    train_pointnet(dataset_folder, excel_path)
