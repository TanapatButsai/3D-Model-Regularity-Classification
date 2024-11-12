import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, 
    recall_score, f1_score, classification_report
)
from sklearn.model_selection import train_test_split
import trimesh
from tqdm import tqdm

# Configuration dictionary with updated settings
config = {
    "dataset_folder": "datasets/hermanmiller/obj-hermanmiller",
    "label_file": "datasets/hermanmiller/label/Final_Validated_Regularity_Levels.xlsx",
    "num_points": 1024,
    "batch_size": 16,
    "num_epochs": 50,          # Increased epochs
    "learning_rate": 0.001,
    "step_size": 15,           # Step size for learning rate scheduler
    "gamma": 0.5,              # Learning rate decay factor
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_classes": 4
}

# Print configuration for reference (for a screenshot)
print("\nConfiguration Settings:")
for key, value in config.items():
    print(f"{key}: {value}")
print("\nMake sure to screenshot this configuration for reference.\n")

# PointNet Model Definition with increased capacity
class PointNet(nn.Module):
    def __init__(self, k=config["num_classes"]):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.conv3 = nn.Conv1d(256, 512, 1)
        self.conv4 = nn.Conv1d(512, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(config["num_points"])
        self.dropout = nn.Dropout(p=0.4)
    
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
def obj_to_pointcloud(obj_path, num_points=config["num_points"]):
    mesh = trimesh.load(obj_path, force='mesh')
    if isinstance(mesh, trimesh.Trimesh):
        points = mesh.sample(num_points)
        return points
    return None

# Process the dataset to prepare point clouds
def process_dataset(dataset_folder, labels_df):
    point_clouds = []
    labels = []
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
        object_id = row['Object ID (Dataset Original Object ID)']
        label = row['Final Regularity Level'] - 1  # Adjust to 0-based labels
        
        obj_file = os.path.join(dataset_folder, object_id.strip(), f"{object_id.strip()}.obj")
        if os.path.isfile(obj_file):
            point_cloud = obj_to_pointcloud(obj_file)
            if point_cloud is not None:
                point_clouds.append(point_cloud)
                labels.append(label)
    return np.array(point_clouds), np.array(labels)

# Train PointNet
def train_pointnet(config):
    labels_df = pd.read_excel(config["label_file"])
    
    # Process dataset
    point_clouds, labels = process_dataset(config["dataset_folder"], labels_df)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        point_clouds, labels, test_size=0.2, random_state=42
    )
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create dataloaders
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # Initialize model, optimizer, and loss function
    device = config["device"]
    model = PointNet(k=config["num_classes"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([3.0, 0.8, 2.0, 1.0]).to(device))  # Adjust class weights
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])
    
    # Training
    model.train()
    for epoch in range(config["num_epochs"]):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {running_loss/len(train_loader):.4f}")
    
    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Metrics with zero_division=1 to suppress warnings
    accuracy = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=1)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=1)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=1)
    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:\n", conf_matrix)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Class 0", "Class 1", "Class 2", "Class 3"], zero_division=1))

if __name__ == "__main__":
    train_pointnet(config)
