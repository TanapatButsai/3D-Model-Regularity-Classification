import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score,
    recall_score, f1_score, roc_auc_score, log_loss
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import trimesh
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Configuration
config = {
    "base_dir": 'datasets/abo/obj-ABO',
    "label_file_path": 'datasets/abo/label/Final_Validated_Regularity_Levels.xlsx',
    "max_data_points": 7800,
    "num_points": 1024,
    "batch_size": 32,
    "num_classes": 5,
    "num_epochs": 100,
    "learning_rate": 0.0005,
    "weight_decay": 1e-5,
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "results_folder": "datasets/abo/results"
}

os.makedirs(config["results_folder"], exist_ok=True)

# Data Augmentation
def augment_pointcloud(points):
    rotation_matrix = trimesh.transformations.random_rotation_matrix()[:3, :3]
    points = np.dot(points, rotation_matrix.T)
    points += np.random.normal(0, 0.02, points.shape)  # Jittering
    scale_factor = np.random.uniform(0.8, 1.2)  # Random scaling
    points *= scale_factor
    return points

# PointNet++ Model
class PointNetPlusPlus(nn.Module):
    def __init__(self, num_classes):
        super(PointNetPlusPlus, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.bn_fc2 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]  # Global max pooling
        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn_fc2(self.fc2(x)))
        x = self.fc3(x)
        return x

# Dataset class for loading and preprocessing data
class PointCloudDataset(Dataset):
    def __init__(self, labels_df, base_dir, num_points):
        self.labels_df = labels_df
        self.base_dir = base_dir
        self.num_points = num_points

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        obj_id = row['Object ID (Dataset Original Object ID)']
        label = int(row['Final Regularity Level']) - 1
        obj_file = os.path.join(self.base_dir, str(obj_id), f'{obj_id.strip()}.obj')

        mesh = trimesh.load(obj_file, force='mesh')
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("Invalid mesh format")

        points = mesh.sample(self.num_points)
        points = augment_pointcloud(points)
        points = torch.tensor(points, dtype=torch.float32)
        return points.T, label

# Load and preprocess data
labels_df = pd.read_excel(config["label_file_path"])
labels_df = labels_df[['Object ID (Dataset Original Object ID)', 'Final Regularity Level']]
labels_df = labels_df.sample(n=min(config["max_data_points"], len(labels_df)), random_state=42)

# Split dataset into training and testing sets
train_df, test_df = train_test_split(labels_df, test_size=0.2, random_state=42, stratify=labels_df['Final Regularity Level'])

# Initialize dataset and dataloader
train_dataset = PointCloudDataset(train_df, config["base_dir"], config["num_points"])
test_dataset = PointCloudDataset(test_df, config["base_dir"], config["num_points"])

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

# Initialize model, loss, and optimizer
model = PointNetPlusPlus(config["num_classes"]).to(config["device"])

# Compute class weights
unique_classes = sorted(train_df['Final Regularity Level'].unique() - 1)
class_weights = np.zeros(config["num_classes"])
for cls in unique_classes:
    class_weights[cls] = 1.0 / (np.sum(train_df['Final Regularity Level'] - 1 == cls) + 1e-6)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(config["device"])

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# Training loop
for epoch in range(config["num_epochs"]):
    model.train()
    running_loss = 0.0
    for points, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}"):
        points, labels = points.to(config["device"]), labels.to(config["device"])
        optimizer.zero_grad()
        outputs = model(points)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    scheduler.step()
    print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {running_loss/len(train_loader):.4f}")

# Evaluation
model.eval()
y_true, y_pred, y_prob = [], [], []

with torch.no_grad():
    for points, labels in test_loader:
        points, labels = points.to(config["device"]), labels.to(config["device"])
        outputs = model(points)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
        y_prob.extend(probabilities.cpu().numpy())

# Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
conf_matrix = confusion_matrix(y_true, y_pred)

# AUC-ROC Calculation with Error Handling
try:
    if len(np.unique(y_true)) > 1:  # Ensure diversity in y_true
        y_true_one_hot = np.zeros((len(y_true), config["num_classes"]))
        y_true_one_hot[np.arange(len(y_true)), y_true] = 1
        auc_roc = roc_auc_score(
            y_true_one_hot,
            np.array(y_prob),
            multi_class='ovr',
            average='weighted',
            labels=np.arange(config["num_classes"])
        )
    else:
        auc_roc = 0.0  # Fallback value if only one class is present
except Exception as e:
    print(f"Error calculating AUC-ROC: {e}")
    auc_roc = 0.0

# Ensure y_prob is normalized (to avoid log_loss warnings)
y_prob = np.array(y_prob)
y_prob /= y_prob.sum(axis=1, keepdims=True)

# Log Loss Calculation with Explicit Class Labels
try:
    logloss = log_loss(
        y_true,
        y_prob,
        labels=np.arange(config["num_classes"])
    )
except ValueError as e:
    print(f"Error calculating Log Loss: {e}")
    logloss = None

# Print Metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"AUC-ROC: {auc_roc:.2f}")
if logloss is not None:
    print(f"Log Loss: {logloss:.4f}")
else:
    print("Log Loss: Calculation skipped due to missing classes.")
print(f"Confusion Matrix:\n{conf_matrix}")

# Save Metrics
metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "AUC-ROC": auc_roc,
    "Log Loss": logloss if logloss is not None else "N/A"
}
pd.DataFrame([metrics]).to_csv(os.path.join(config["results_folder"], "metrics.csv"), index=False)

# Save Confusion Matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap="Blues", interpolation="nearest")
plt.colorbar()
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.savefig(os.path.join(config["results_folder"], "confusion_matrix.png"))
plt.close()
