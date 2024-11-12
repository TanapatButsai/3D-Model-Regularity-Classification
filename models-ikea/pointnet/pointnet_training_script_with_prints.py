import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import trimesh
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score,
                             f1_score, roc_auc_score, log_loss)
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# Function to load point cloud from an OBJ file, ensuring only meshes are used
def load_pointcloud_from_obj(file_path, num_points=1024):
    try:
        mesh = trimesh.load(file_path, force='mesh')  # Force loading as a mesh
        if isinstance(mesh, trimesh.Trimesh):
            points = mesh.sample(num_points)  # Sample points if it's a valid mesh
            return points
        else:
            print(f"Skipping non-mesh file {file_path}")
            return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Process the dataset to extract point clouds and corresponding labels
def process_dataset_structure(base_folder, labels):
    print("Starting data extraction...")
    point_clouds = []
    targets = []

    for index, row in tqdm(labels.iterrows(), total=len(labels)):
        object_id = row['Object ID (Dataset Original Object ID)']
        regularity_level = row['Final Regularity Level']
        folder_name = row['FolderName']

        obj_folder = os.path.join(base_folder, folder_name.strip(), object_id.strip())
        obj_file = os.path.join(obj_folder, 'ikea_model.obj')

        if os.path.isfile(obj_file):
            point_cloud = load_pointcloud_from_obj(obj_file)
            if point_cloud is not None:
                point_clouds.append(point_cloud)
                targets.append(regularity_level)
        else:
            print(f"OBJ file not found: {obj_file}")

    print("Data extraction completed.")
    return np.array(point_clouds), np.array(targets)

# Paths to your dataset folder and Excel file
base_folder = 'datasets/IKEA/obj-IKEA'
label_file = 'datasets/IKEA/label/Final_Validated_Regularity_Levels.xlsx'

# Load labels from the Excel file
print("Loading labels from Excel file...")
label_data = pd.read_excel(label_file)
print("Labels loaded.")

# Process the dataset to extract point clouds and labels
point_clouds, targets = process_dataset_structure(base_folder, label_data)

if len(point_clouds) == 0:
    print("No point clouds loaded. Please check the dataset and file paths.")
    exit()

# Prepare data
print("Preparing data...")
X = np.array(point_clouds)
y = np.array(targets) - np.min(targets)  # Normalize labels to start from 0
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Split dataset into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# PointNet model definition
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

# Initialize model, optimizer, and loss function
print("Initializing model, optimizer, and loss function...")
model = PointNet(k=len(np.unique(y)))  # Set output classes based on unique labels in y
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training setup
num_epochs = 10
batch_size = 4
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Training loop
print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.transpose(2, 1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

print("Training completed.")

# Evaluation and metrics calculation
print("Starting evaluation...")
model.eval()
all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.transpose(2, 1)
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)  # Softmax to get probabilities
        _, predicted = torch.max(outputs, 1)
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# Calculate metrics
print("Calculating metrics...")
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

accuracy = accuracy_score(all_labels, all_preds)
conf_matrix = confusion_matrix(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')
auc_roc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
log_loss_val = log_loss(all_labels, all_probs)

# Print metrics
print(f"Test Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"AUC-ROC: {auc_roc:.2f}")
print(f"Log Loss: {log_loss_val:.4f}")
