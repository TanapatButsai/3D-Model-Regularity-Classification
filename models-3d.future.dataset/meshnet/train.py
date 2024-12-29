import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score,
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
    "base_dir": 'datasets/3d-future-dataset/obj-3d.future',
    "label_file_path": 'datasets/3d-future-dataset/label/Final_Validated_Regularity_Levels.xlsx',
    "max_data_points": 10000,
    "num_points": 1024,
    "batch_size": 32, 
    "num_classes": 4,
    "num_epochs": 100,
    "learning_rate": 0.0005,
    "weight_decay": 1e-5,
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "conf_matrix_path": "confusion_matrix.png",
}

# Early Stopping
class EarlyStopping:
    def __init__(self, patience=10, delta=0, path="checkpoint.pt", verbose=False):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        """Save model when validation loss improves."""
        if self.verbose:
            print("Validation loss improved. Saving model...")
        torch.save(model.state_dict(), self.path)

# Data Augmentation
def augment_mesh(vertices, normals):
    rotation_matrix = trimesh.transformations.random_rotation_matrix()[:3, :3]
    vertices = np.dot(vertices, rotation_matrix.T)
    jitter = np.random.normal(0, 0.02, vertices.shape)
    vertices += jitter
    scale_factor = np.random.uniform(0.8, 1.2)
    vertices *= scale_factor
    return vertices, normals

# MeshNet Model
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
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, vertices, normals):
        x = torch.cat([vertices, normals], dim=2).transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = torch.max(x, dim=-1)[0]  # Global Max Pooling
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Dataset Class
class MeshDataset(Dataset):
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
        obj_file = os.path.join(self.base_dir, str(obj_id), 'normalized_model.obj')

        mesh = trimesh.load(obj_file, force='mesh')
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("Invalid mesh format")
        vertices, normals = mesh.vertices, mesh.vertex_normals
        if len(vertices) > self.num_points:
            indices = np.random.choice(len(vertices), self.num_points, replace=False)
            vertices, normals = vertices[indices], normals[indices]
        elif len(vertices) < self.num_points:
            padding = self.num_points - len(vertices)
            vertices = np.vstack([vertices, np.zeros((padding, 3))])
            normals = np.vstack([normals, np.zeros((padding, 3))])

        vertices, normals = augment_mesh(vertices, normals)
        return {
            'vertices': torch.tensor(vertices, dtype=torch.float32),
            'normals': torch.tensor(normals, dtype=torch.float32),
        }, torch.tensor(label, dtype=torch.long)

# Load Data
labels_df = pd.read_excel(config["label_file_path"])
labels_df = labels_df[['Object ID (Dataset Original Object ID)', 'Final Regularity Level']]
labels_df = labels_df.sample(n=min(config["max_data_points"], len(labels_df)), random_state=42)

# Train-Test Split
train_df, test_df = train_test_split(
    labels_df, test_size=0.2, random_state=42, stratify=labels_df['Final Regularity Level']
)

# DataLoaders
train_dataset = MeshDataset(train_df, config["base_dir"], config["num_points"])
test_dataset = MeshDataset(test_df, config["base_dir"], config["num_points"])

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

# Initialize Model, Loss, Optimizer, and Scheduler
model = MeshNet(config["num_classes"]).to(config["device"])
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# Early Stopping
early_stopping = EarlyStopping(patience=10, delta=0.01, path="best_model.pt", verbose=True)

# Training Loop
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

    scheduler.step()
    val_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")

    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping triggered. Training halted.")
        break

# Load Best Model
model.load_state_dict(torch.load("best_model.pt"))

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
auc_roc = roc_auc_score(
    np.eye(config["num_classes"])[y_true], np.array(y_prob), multi_class='ovr', average='weighted'
)
logloss = log_loss(y_true, y_prob)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"AUC-ROC: {auc_roc:.2f}")
print(f"Log Loss: {logloss:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", conf_matrix)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(config["conf_matrix_path"])
plt.close()
