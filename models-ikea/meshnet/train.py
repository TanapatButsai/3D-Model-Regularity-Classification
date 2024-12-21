import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from tqdm import tqdm
import trimesh
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.exceptions import UndefinedMetricWarning
import warnings

# Suppress warnings for undefined metrics
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Configuration
config = {
    "base_dir": 'datasets/IKEA/obj-IKEA',
    "label_file_path": 'datasets/IKEA/label/Final_Validated_Regularity_Levels.xlsx',
    "num_points": 1024,
    "batch_size": 32,
    "num_classes": 4,
    "num_epochs": 70,
    "learning_rate": 0.0005,
    "weight_decay": 1e-4,
    "patience": 5,
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
}

def print_config(config):
    print("Configuration Settings:")
    for key, value in config.items():
        print(f"{key}: {value}")

print_config(config)

# Extract mesh features (vertices, normals)
def extract_mesh_features(file_path, max_vertices=1024):
    try:
        mesh = trimesh.load(file_path)
        if isinstance(mesh, trimesh.Scene):
            if mesh.geometry:
                mesh = trimesh.util.concatenate(mesh.geometry.values())
            else:
                return None, None
        if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'vertex_normals'):
            return None, None

        vertices = np.array(mesh.vertices, dtype=np.float32)
        normals = np.array(mesh.vertex_normals, dtype=np.float32)

        # Pad or truncate
        if vertices.shape[0] > max_vertices:
            indices = np.random.choice(vertices.shape[0], max_vertices, replace=False)
            vertices, normals = vertices[indices], normals[indices]
        elif vertices.shape[0] < max_vertices:
            padding = np.zeros((max_vertices - vertices.shape[0], 3), dtype=np.float32)
            vertices = np.vstack([vertices, padding])
            normals = np.vstack([normals, padding])

        return vertices, normals
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None

# Dataset class
class MeshDataset(Dataset):
    def __init__(self, labels_df, base_dir, max_vertices=1024, augment=False):
        self.labels_df = labels_df
        self.base_dir = base_dir
        self.max_vertices = max_vertices
        self.augment = augment

    def augment_vertices(self, vertices):
        theta = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        vertices = np.dot(vertices, rotation_matrix)
        scale = np.random.uniform(0.9, 1.1)
        vertices *= scale
        jitter = np.random.normal(0, 0.02, size=vertices.shape)
        vertices += jitter
        return vertices

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        obj_id = row['Object ID (Dataset Original Object ID)']
        folder_name = row['FolderName']
        label = int(row['Final Regularity Level']) - 1
        obj_file = os.path.join(self.base_dir, folder_name.strip(), obj_id.strip(), 'ikea_model.obj')

        vertices, normals = extract_mesh_features(obj_file, self.max_vertices)
        if vertices is None or normals is None:
            vertices = np.zeros((self.max_vertices, 3), dtype=np.float32)
            normals = np.zeros((self.max_vertices, 3), dtype=np.float32)
        elif self.augment:
            vertices = self.augment_vertices(vertices)

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

# Training function
def train_meshnet(config):
    labels_df = pd.read_excel(config["label_file_path"])
    train_df, test_df = train_test_split(labels_df, test_size=0.2, random_state=42, stratify=labels_df['Final Regularity Level'])

    train_dataset = MeshDataset(train_df, config["base_dir"], augment=True)
    test_dataset = MeshDataset(test_df, config["base_dir"], augment=False)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    model = MeshNet(config["num_classes"]).to(config["device"])
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    class_counts = train_df['Final Regularity Level'].value_counts()
    class_weights = torch.tensor(1.0 / class_counts.values, dtype=torch.float32).to(config["device"])
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_loss = float('inf')
    patience = 0

    for epoch in range(config["num_epochs"]):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}"):
            inputs, labels = batch
            vertices, normals = inputs['vertices'].to(config["device"]), inputs['normals'].to(config["device"])
            labels = labels.to(config["device"])
            optimizer.zero_grad()
            outputs = model(vertices, normals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
        else:
            patience += 1
            if patience >= config["patience"]:
                print("Early stopping triggered.")
                break

    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            vertices, normals = inputs['vertices'].to(config["device"]), inputs['normals'].to(config["device"])
            labels = labels.to(config["device"])
            outputs = model(vertices, normals)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
    conf_matrix = confusion_matrix(y_true, y_pred)

    try:
        auc_roc = roc_auc_score(np.eye(config["num_classes"])[y_true], y_prob, multi_class='ovr')
    except ValueError:
        auc_roc = None

    try:
        logloss = log_loss(y_true, y_prob, labels=np.arange(config["num_classes"]))
    except ValueError as e:
        logloss = None
        print(f"Log Loss Error: {e}")

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"AUC-ROC: {auc_roc:.2f}" if auc_roc is not None else "AUC-ROC: None")
    print(f"Log Loss: {logloss:.4f}" if logloss is not None else "Log Loss: None")
    print(f"Confusion Matrix:\n{conf_matrix}")

train_meshnet(config)
