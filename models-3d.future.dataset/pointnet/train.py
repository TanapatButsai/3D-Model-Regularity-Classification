import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from tqdm import tqdm
import trimesh
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration settings
config = {
    "base_dir": 'datasets/3d-future-dataset/obj-3d.future',
    "label_file_path": 'datasets/3d-future-dataset/label/Final_Validated_Regularity_Levels.xlsx',
    "max_data_points": 10000,
    "point_cloud_size": 1024,
    "batch_size": 32,
    "num_classes": 4,
    "num_epochs": 100,
    "learning_rate": 0.001,
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "patience": 10  # For early stopping
}

print("Configuration Settings:")
for key, value in config.items():
    print(f"{key}: {value}")

# PointNet Model
class PointNet(nn.Module):
    def __init__(self, num_classes):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = torch.relu(self.bn4(self.fc1(x)))
        x = torch.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)

# Dataset class for loading and preprocessing data
class PointCloudDataset(Dataset):
    def __init__(self, labels_df, base_dir, point_cloud_size=1024):
        self.labels_df = labels_df
        self.base_dir = base_dir
        self.point_cloud_size = point_cloud_size

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        obj_id = row['Object ID (Dataset Original Object ID)']
        label = int(row['Final Regularity Level']) - 1  # Adjust label to start from 0
        obj_file = os.path.join(self.base_dir, str(obj_id), 'normalized_model.obj')

        mesh = trimesh.load(obj_file, force='mesh')
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("Invalid mesh format")

        points = mesh.sample(self.point_cloud_size)
        points = torch.tensor(points, dtype=torch.float32)
        return points, label

# Load and preprocess data
labels_df = pd.read_excel(config["label_file_path"])
labels_df = labels_df[['Object ID (Dataset Original Object ID)', 'Final Regularity Level']]
labels_df = labels_df.sample(n=min(config["max_data_points"], len(labels_df)), random_state=42)

# Initialize dataset and dataloader
dataset = PointCloudDataset(labels_df, config["base_dir"], config["point_cloud_size"])
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

# Initialize model, loss, optimizer
model = PointNet(num_classes=config["num_classes"]).to(config["device"])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# Training with early stopping
best_loss = float('inf')
epochs_no_improve = 0
for epoch in range(config["num_epochs"]):
    model.train()
    running_loss = 0.0
    y_true, y_pred = [], []

    for data, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}"):
        data, labels = data.to(config["device"]), labels.to(config["device"])
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Epoch [{epoch+1}/{config['num_epochs']}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

    # Early stopping
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= config["patience"]:
            print("Early stopping triggered. Restoring best model.")
            model.load_state_dict(torch.load('best_model.pth'))
            break

# Evaluation
model.eval()
y_true, y_pred, y_probs = [], [], []
with torch.no_grad():
    for data, labels in dataloader:
        data, labels = data.to(config["device"]), labels.to(config["device"])
        outputs = model(data)
        y_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
auc_roc = roc_auc_score(y_true, y_probs, multi_class='ovr')
logloss = log_loss(y_true, y_probs)

# Print metrics
print("Final Evaluation Results:")
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"AUC-ROC: {auc_roc:.2f}")
print(f"Log Loss: {logloss:.4f}")

print("\nConfiguration at End of Training:")
for key, value in config.items():
    print(f"{key}: {value}")
