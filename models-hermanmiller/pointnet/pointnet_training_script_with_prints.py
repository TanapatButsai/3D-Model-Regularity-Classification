
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import trimesh
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

# PointNet Model Definition with increased capacity
class PointNet(nn.Module):
    def __init__(self, k=4):
        super(PointNet, self).__init__()
        
        self.conv1 = nn.Conv1d(3, 128, 1)  # Increased layer size
        self.conv2 = nn.Conv1d(128, 256, 1)  # Increased layer size
        self.conv3 = nn.Conv1d(256, 512, 1)  # Increased layer size
        self.conv4 = nn.Conv1d(512, 1024, 1)  # Added extra layer
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(1024)
        self.dropout = nn.Dropout(p=0.4)  # Increased dropout to reduce overfitting
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        
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

# Apply advanced data augmentation: jittering, random flipping, rotation, and scaling
def augment_pointcloud(points):
    # Apply random jittering (adding small noise)
    jitter = np.random.normal(0, 0.02, size=points.shape)
    points += jitter
    
    # Apply random rotation around the z-axis
    theta = np.random.uniform(0, 2*np.pi)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta),  np.cos(theta), 0],
                                [0,              0,             1]])
    points = np.dot(points, rotation_matrix.T)
    
    # Apply random scaling
    scale = np.random.uniform(0.8, 1.2)
    points *= scale
    
    # Apply random flipping along the x-axis
    if np.random.rand() > 0.5:
        points[:, 0] = -points[:, 0]
    
    return points

# Process the dataset to convert OBJ files into point clouds
def process_dataset(dataset_folder, excel_data, num_points=1024):
    print("Starting dataset extraction and point cloud conversion...")
    point_clouds = []
    labels = []
    
    for index, row in excel_data.iterrows():
        object_id = row['Object ID (Dataset Original Object ID)']
        label = row['Final Regularity Level'] - 1  # Adjust label to be 0-based
        
        obj_folder = os.path.join(dataset_folder, object_id)
        obj_file = os.path.join(obj_folder, f"{object_id}.obj")
        
        if os.path.exists(obj_file):
            print(f"Processing {object_id}...")
            point_cloud = obj_to_pointcloud(obj_file, num_points=num_points)
            
            # Apply advanced data augmentation
            point_cloud = augment_pointcloud(point_cloud)
            
            point_clouds.append(point_cloud)
            labels.append(label)
    
    print("Finished dataset processing.")
    return np.array(point_clouds), np.array(labels)

# Main training loop
def train_pointnet(dataset_folder, excel_path, num_epochs=30):  # Increased to 30 epochs
    print("Loading Excel data...")
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
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)  # Increased batch size
    
    # Class weighting based on class frequencies
    class_weights = torch.tensor([3.0, 0.8, 2.0, 1.0], dtype=torch.float32).to('cpu')  # Fine-tuned weights
    
    # Initialize model, optimizer, and loss function
    print("Initializing PointNet model...")
    device = torch.device("cpu")
    model = PointNet(k=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Reduce LR every 10 epochs
    
    print("Starting training...")
    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, target in dataloader:
            inputs = inputs.transpose(2, 1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
        scheduler.step()  # Update learning rate
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    print("Training completed!")
    
    # Evaluation phase
    print("Starting evaluation...")
    model.eval()  # Set the model to evaluation mode
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.transpose(2, 1)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Classification report
    report = classification_report(all_labels, all_preds, target_names=["Class 0", "Class 1", "Class 2", "Class 3"])
    print("Classification Report:")
    print(report)
    
# Example usage
if __name__ == "__main__":
    dataset_folder = "datasets/hermanmiller/obj-hermanmiller"
    excel_path = "datasets/hermanmiller/label/Final_Validated_Regularity_Levels.xlsx"
    train_pointnet(dataset_folder, excel_path)
