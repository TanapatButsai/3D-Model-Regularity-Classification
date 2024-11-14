import os
import pandas as pd
import numpy as np
import trimesh
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import warnings

# Configuration dictionary
config = {
    "base_dir": 'datasets/3d-future-dataset/obj-3d.future',
    "label_file_path": "datasets/3d-future-dataset/label/Final_Validated_Regularity_Levels.xlsx",
    "max_data_points": 10000,
    "point_cloud_size": 1024,
    "batch_size": 32,
    "num_classes": 4,
    "num_epochs": 100,
    "learning_rate": 0.001,
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "patience": 10  # Number of epochs with no improvement before stopping
}

# Print the configuration settings at the start
print("Configuration Settings:")
for key, value in config.items():
    print(f"{key}: {value}")
print("\nStarting Training...")

# Load label data and preprocess
labels_df = pd.read_excel(config["label_file_path"])
labels_df = labels_df[['Object ID (Dataset Original Object ID)', 'Final Regularity Level']]
labels_df = labels_df.sample(n=min(config["max_data_points"], len(labels_df)), random_state=42)
labels_df['Final Regularity Level'] -= 1  # Adjust labels to start from 0

# Point cloud extraction function from OBJ files
def extract_point_cloud(obj_file):
    try:
        mesh = trimesh.load(obj_file, force='mesh')
        if not isinstance(mesh, trimesh.Trimesh):
            return None  # Skip non-mesh files
        points = mesh.vertices  # Extract vertices as point cloud
        return points
    except Exception as e:
        print(f"Error processing file {obj_file}: {e}")
        return None

# Prepare point cloud dataset and corresponding labels
class PointCloudDataset(Dataset):
    def __init__(self, labels_df, base_dir, point_cloud_size):
        self.labels_df = labels_df
        self.base_dir = base_dir
        self.point_cloud_size = point_cloud_size
        self.point_clouds = []
        self.labels = []

        for index, row in tqdm(self.labels_df.iterrows(), total=len(self.labels_df), desc="Loading Point Clouds"):
            obj_id = row['Object ID (Dataset Original Object ID)']
            obj_file = os.path.join(base_dir, str(obj_id), 'normalized_model.obj')
            if os.path.exists(obj_file):
                points = extract_point_cloud(obj_file)
                if points is not None:
                    # Sample or pad points to configured point cloud size
                    if len(points) > self.point_cloud_size:
                        points = points[:self.point_cloud_size]
                    elif len(points) < self.point_cloud_size:
                        points = np.pad(points, ((0, self.point_cloud_size - len(points)), (0, 0)), mode='constant')
                    self.point_clouds.append(points)
                    self.labels.append(row['Final Regularity Level'])

        self.point_clouds = np.array(self.point_clouds)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        return torch.tensor(self.point_clouds[idx], dtype=torch.float32), self.labels[idx]

# Load the dataset
dataset = PointCloudDataset(labels_df, config["base_dir"], config["point_cloud_size"])
X_train, X_test, y_train, y_test = train_test_split(dataset.point_clouds, dataset.labels, test_size=0.2, random_state=42)

train_dataset = [(torch.tensor(X, dtype=torch.float32).to(config["device"]), y) for X, y in zip(X_train, y_train)]
test_dataset = [(torch.tensor(X, dtype=torch.float32).to(config["device"]), y) for X, y in zip(X_test, y_test)]

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

# Define PointNet model structure
class PointNet(nn.Module):
    # Add PointNet structure here
    def __init__(self, num_classes=4):
        super(PointNet, self).__init__()
        # Define PointNet layers (use existing PyTorch PointNet implementations if available)
        pass

    def forward(self, x):
        # Forward pass logic
        pass

# Initialize PointNet
model = PointNet(num_classes=config["num_classes"]).to(config["device"])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

# Early stopping parameters
best_val_loss = float('inf')
epochs_no_improve = 0

# Train PointNet model
for epoch in range(config["num_epochs"]):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(config["device"]), labels.to(config["device"])
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Calculate validation loss for early stopping
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(config["device"]), labels.to(config["device"])
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(test_loader)
    print(f"Epoch [{epoch+1}/{config['num_epochs']}], Train Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

    # Early stopping condition
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= config["patience"]:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss for {config['patience']} epochs.")
            break

# Evaluate model on test set
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(config["device"]), labels.to(config["device"])
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
recall = recall_score(all_labels, all_preds, average='weighted', zero_division=1)
f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)
auc_roc = roc_auc_score(all_labels, torch.nn.functional.softmax(torch.tensor(all_preds), dim=1), multi_class='ovr')
logloss = log_loss(all_labels, torch.nn.functional.softmax(torch.tensor(all_preds), dim=1))

print("\nTraining Complete.")
print("\nFinal Configuration Settings:")
for key, value in config.items():
    print(f"{key}: {value}")

# Print final results
print("\nPointNet Results:")
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"AUC-ROC: {auc_roc}")
print(f"Log Loss: {logloss}")
