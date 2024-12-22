import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss
)
from torch.utils.data import DataLoader, Dataset
import trimesh
from tqdm import tqdm
import torch.nn.functional as F

# Configuration
config = {
    "label_file": "datasets/pix3d/label/Final_Validated_Regularity_Levels.xlsx",
    "obj_folder": "datasets/pix3d/obj-pix3d",
    "num_points": 1024,
    "batch_size": 16,
    "num_epochs": 60,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# Load labels from Excel
label_data = pd.read_excel(config["label_file"])
labels = label_data[['Object ID (Dataset Original Object ID)', 'Final Regularity Level']]

# Filter classes with at least 2 samples
class_counts = labels['Final Regularity Level'].value_counts()
valid_classes = class_counts[class_counts >= 2].index
labels = labels[labels['Final Regularity Level'].isin(valid_classes)]

# Encode labels and calculate num_classes
label_encoder = LabelEncoder()
labels['Final Regularity Level'] = label_encoder.fit_transform(labels['Final Regularity Level'])
num_classes = len(label_encoder.classes_)

# Train-Test Split
train_df, test_df = train_test_split(
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels['Final Regularity Level']
)

# Function to load mesh data
def load_mesh(file_path, num_points):
    try:
        mesh = trimesh.load(file_path)
        if isinstance(mesh, trimesh.Scene) and len(mesh.geometry) > 0:
            mesh = list(mesh.geometry.values())[0]
        if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'vertex_normals'):
            return None, None
        vertices = mesh.vertices
        normals = mesh.vertex_normals

        # Sample or pad vertices and normals
        if len(vertices) > num_points:
            indices = np.random.choice(len(vertices), num_points, replace=False)
            vertices = vertices[indices]
            normals = normals[indices]
        elif len(vertices) < num_points:
            padding = num_points - len(vertices)
            vertices = np.vstack([vertices, np.zeros((padding, 3))])
            normals = np.vstack([normals, np.zeros((padding, 3))])

        return vertices, normals
    except Exception as e:
        print(f"Error loading mesh from {file_path}: {e}")
        return None, None

# Dataset class
class MeshDataset(Dataset):
    def __init__(self, dataframe, obj_folder, num_points, augment=False):
        self.dataframe = dataframe
        self.obj_folder = obj_folder
        self.num_points = num_points
        self.augment = augment

    def augment_data(self, vertices):
        theta = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        vertices = np.dot(vertices, rotation_matrix)
        scale = np.random.uniform(0.9, 1.1)
        vertices *= scale
        jitter = np.random.normal(0, 0.02, vertices.shape)
        vertices += jitter
        return vertices

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        obj_id = row['Object ID (Dataset Original Object ID)']
        label = int(row['Final Regularity Level'])
        file_path = os.path.join(self.obj_folder, "models", obj_id.strip(), "model.obj")

        vertices, normals = load_mesh(file_path, self.num_points)
        if vertices is None or normals is None:
            vertices = np.zeros((self.num_points, 3))
            normals = np.zeros((self.num_points, 3))
        elif self.augment:
            vertices = self.augment_data(vertices)

        return {
            'vertices': torch.tensor(vertices, dtype=torch.float32),
            'normals': torch.tensor(normals, dtype=torch.float32),
        }, torch.tensor(label, dtype=torch.long)

# MeshNet model
class MeshNet(nn.Module):
    def __init__(self, num_classes):
        super(MeshNet, self).__init__()
        self.conv1 = nn.Conv1d(6, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, vertices, normals):
        x = torch.cat([vertices, normals], dim=2).transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = torch.max(x, dim=-1)[0]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model, loss, and optimizer
train_dataset = MeshDataset(train_df, config["obj_folder"], config["num_points"], augment=True)
test_dataset = MeshDataset(test_df, config["obj_folder"], config["num_points"], augment=False)

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

model = MeshNet(num_classes).to(config["device"])
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(config["num_epochs"]):
    model.train()
    running_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}"):
        inputs, labels = batch
        vertices = inputs['vertices'].to(config["device"])
        normals = inputs['normals'].to(config["device"])
        labels = labels.to(config["device"])

        optimizer.zero_grad()
        outputs = model(vertices, normals)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {running_loss/len(train_loader):.4f}")

# Evaluation
model.eval()
y_true, y_pred, y_prob = [], [], []
with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch
        vertices = inputs['vertices'].to(config["device"])
        normals = inputs['normals'].to(config["device"])
        labels = labels.to(config["device"])

        outputs = model(vertices, normals)
        probs = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
        y_prob.extend(probs.cpu().numpy())

# Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
conf_matrix = confusion_matrix(y_true, y_pred)

try:
    auc_roc = roc_auc_score(np.eye(len(np.unique(y_true)))[y_true], y_prob, multi_class='ovr')
except ValueError:
    auc_roc = None

logloss = log_loss(y_true, y_prob)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"AUC-ROC: {auc_roc:.2f}" if auc_roc is not None else "AUC-ROC: None")
print(f"Log Loss: {logloss:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")
