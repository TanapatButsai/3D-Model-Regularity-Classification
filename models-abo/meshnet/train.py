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

# Configuration
config = {
    "base_dir": 'datasets/abo/obj-ABO',
    "label_file_path": 'datasets/abo/label/Final_Validated_Regularity_Levels.xlsx',
    # "max_data_points": 500,
    "max_data_points": 7800,
    "num_points": 1024,
    "batch_size": 32,
    "num_classes": 5,
    "num_epochs": 50,
    "learning_rate": 0.0005,
    "weight_decay": 1e-5,
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "results_folder": "datasets/abo/results"
}

os.makedirs(config["results_folder"], exist_ok=True)

# Data Augmentation
def augment_mesh(vertices):
    theta = np.random.uniform(0, 2 * np.pi)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    vertices = np.dot(vertices, rotation_matrix)
    vertices *= np.random.uniform(0.8, 1.2)  # Scaling
    vertices += np.random.normal(0, 0.02, vertices.shape)  # Jittering
    return vertices

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
        x = torch.max(x, dim=-1)[0]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Dataset Class
class MeshDataset(Dataset):
    def __init__(self, labels_df, base_dir, num_points, augment=False):
        self.labels_df = labels_df
        self.base_dir = base_dir
        self.num_points = num_points
        self.augment = augment

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        obj_id = row['Object ID (Dataset Original Object ID)']
        label = int(row['Final Regularity Level']) - 1
        obj_file = os.path.join(self.base_dir, str(obj_id), f'{obj_id.strip()}.obj')

        mesh = trimesh.load(obj_file, force='mesh')
        vertices, normals = mesh.vertices, mesh.vertex_normals

        if len(vertices) > self.num_points:
            indices = np.random.choice(len(vertices), self.num_points, replace=False)
            vertices, normals = vertices[indices], normals[indices]
        elif len(vertices) < self.num_points:
            padding = self.num_points - len(vertices)
            vertices = np.vstack([vertices, np.zeros((padding, 3))])
            normals = np.vstack([normals, np.zeros((padding, 3))])

        if self.augment:
            vertices = augment_mesh(vertices)

        return {
            'vertices': torch.tensor(vertices, dtype=torch.float32),
            'normals': torch.tensor(normals, dtype=torch.float32),
        }, torch.tensor(label, dtype=torch.long)

# Data Loading
labels_df = pd.read_excel(config["label_file_path"])
labels_df = labels_df.sample(n=min(config["max_data_points"], len(labels_df)), random_state=42)

train_df, test_df = train_test_split(labels_df, test_size=0.2, random_state=42, stratify=labels_df['Final Regularity Level'])

train_dataset = MeshDataset(train_df, config["base_dir"], config["num_points"], augment=True)
test_dataset = MeshDataset(test_df, config["base_dir"], config["num_points"], augment=False)

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

# Initialize Model, Optimizer, and Scheduler
model = MeshNet(config["num_classes"]).to(config["device"])

class_weights = torch.tensor(
    [1.0 / max(1, len(train_df[train_df['Final Regularity Level'] == i])) for i in range(1, config["num_classes"] + 1)],
    dtype=torch.float32
).to(config["device"])

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=10, delta=0, path="checkpoint.pt", verbose=False):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, metric, model):
        score = -metric
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        if self.verbose:
            print("Saving model...")
        torch.save(model.state_dict(), self.path)

early_stopping = EarlyStopping(patience=10, path=os.path.join(config["results_folder"], "best_model.pt"), verbose=True)

# Training Loop
for epoch in range(config["num_epochs"]):
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}"):
        inputs, labels = batch
        vertices, normals = inputs['vertices'].to(config["device"]), inputs['normals'].to(config["device"])
        labels = labels.to(config["device"])

        optimizer.zero_grad()
        outputs = model(vertices, normals)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    scheduler.step()
    train_loss /= len(train_loader)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")

    # Validation Phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            vertices, normals = inputs['vertices'].to(config["device"]), inputs['normals'].to(config["device"])
            labels = labels.to(config["device"])

            outputs = model(vertices, normals)
            val_loss += criterion(outputs, labels).item()

    val_loss /= len(test_loader)
    print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")

    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping triggered.")
        break

# Evaluation
model.load_state_dict(torch.load(os.path.join(config["results_folder"], "best_model.pt")))
model.eval()
y_true, y_pred, y_prob = [], [], []

with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch
        vertices, normals = inputs['vertices'].to(config["device"]), inputs['normals'].to(config["device"])
        labels = labels.to(config["device"])

        outputs = model(vertices, normals)
        probs = F.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs.cpu().numpy())

# Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
conf_matrix = confusion_matrix(y_true, y_pred)

try:
    auc_roc = roc_auc_score(np.eye(config["num_classes"])[y_true], y_prob, multi_class='ovr')
except Exception as e:
    print(f"AUC-ROC Error: {e}")
    auc_roc = None

try:
    logloss = log_loss(y_true, y_prob)
except Exception as e:
    print(f"Log Loss Error: {e}")
    logloss = None

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC-ROC: {auc_roc:.4f}" if auc_roc is not None else "AUC-ROC: None")
print(f"Log Loss: {logloss:.4f}" if logloss is not None else "Log Loss: None")
print(f"Confusion Matrix:\n{conf_matrix}")
