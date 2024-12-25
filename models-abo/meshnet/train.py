import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
import trimesh
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Configuration
config = {
    "dataset_folder": "datasets/abo/obj-ABO",
    "label_file": "datasets/abo/label/Final_Validated_Regularity_Levels.xlsx",
    "num_points": 1024,
    "batch_size": 32,
    "num_epochs": 80,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_classes": 5,
    "conf_matrix_path": "datasets/abo/conf_matrix.png",
    "max_data_points": 7800
}

# Load 3D mesh and extract features
def load_mesh(file_path, num_points):
    try:
        mesh = trimesh.load(file_path, force='mesh')
        if isinstance(mesh, trimesh.Trimesh):
            vertices = mesh.vertices
            normals = mesh.vertex_normals
            if len(vertices) > num_points:
                indices = np.random.choice(len(vertices), num_points, replace=False)
                vertices, normals = vertices[indices], normals[indices]
            elif len(vertices) < num_points:
                padding = num_points - len(vertices)
                vertices = np.vstack([vertices, np.zeros((padding, 3))])
                normals = np.vstack([normals, np.zeros((padding, 3))])
            return vertices, normals
    except Exception as e:
        print(f"Error loading mesh: {file_path}: {e}")
    return None, None

# Data augmentation
def augment_mesh(vertices, normals):
    rotation_matrix = trimesh.transformations.random_rotation_matrix()[:3, :3]
    vertices = np.dot(vertices, rotation_matrix.T)
    jitter = np.random.normal(0, 0.02, vertices.shape)
    vertices += jitter
    return vertices, normals

# Data preprocessing
def preprocess_data(config):
    labels_df = pd.read_excel(config["label_file"])
    labels_df = labels_df.sample(n=min(config["max_data_points"], len(labels_df)), random_state=42)
    point_clouds, normals, labels = [], [], []

    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
        object_id = row['Object ID (Dataset Original Object ID)']
        label = row['Final Regularity Level'] - 1
        file_path = os.path.join(config["dataset_folder"], object_id.strip(), f"{object_id.strip()}.obj")

        if os.path.isfile(file_path):
            vertices, normals_data = load_mesh(file_path, config["num_points"])
            if vertices is not None:
                vertices, normals_data = augment_mesh(vertices, normals_data)
                point_clouds.append(vertices)
                normals.append(normals_data)
                labels.append(label)

    return train_test_split(
        torch.tensor(np.array(point_clouds), dtype=torch.float32),
        torch.tensor(np.array(normals), dtype=torch.float32),
        torch.tensor(np.array(labels), dtype=torch.long),
        test_size=0.2,
        random_state=42
    )

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
        self.dropout = nn.Dropout(p=0.3)

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

# Train the model
def train_model(config):
    X_train, X_test, N_train, N_test, y_train, y_test = preprocess_data(config)
    train_loader = DataLoader(
        torch.utils.data.TensorDataset(X_train, N_train, y_train), batch_size=config["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        torch.utils.data.TensorDataset(X_test, N_test, y_test), batch_size=config["batch_size"], shuffle=False
    )
    
    model = MeshNet(config["num_classes"]).to(config["device"])
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    for epoch in range(config["num_epochs"]):
        model.train()
        running_loss = 0.0
        for vertices, normals, labels in train_loader:
            vertices, normals, labels = (
                vertices.to(config["device"]),
                normals.to(config["device"]),
                labels.to(config["device"]),
            )
            optimizer.zero_grad()
            outputs = model(vertices, normals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {running_loss/len(train_loader):.4f}")

    evaluate_model(model, test_loader, config)

# Evaluate the model
def evaluate_model(model, test_loader, config):
    import matplotlib.pyplot as plt  # Ensure matplotlib is imported
    from sklearn.preprocessing import LabelBinarizer

    model.eval()
    all_preds, all_labels, all_probas = [], [], []

    with torch.no_grad():
        for vertices, normals, labels in test_loader:
            vertices, normals, labels = (
                vertices.to(config["device"]),
                normals.to(config["device"]),
                labels.to(config["device"]),
            )
            outputs = model(vertices, normals)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probas.extend(probs.cpu().numpy())

    # Metrics Calculation
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=1)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=1)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=1)

    # Adjust for AUC-ROC Calculation
    lb = LabelBinarizer()
    lb.fit(range(config["num_classes"]))  # Ensure it includes all classes
    all_labels_bin = lb.transform(all_labels)  # One-hot encode true labels

    try:
        auc_roc = roc_auc_score(all_labels_bin, np.array(all_probas), multi_class="ovr", average="weighted")
    except ValueError as e:
        print(f"Error calculating AUC-ROC: {e}")
        auc_roc = None

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"AUC-ROC: {auc_roc:.2f}" if auc_roc is not None else "AUC-ROC: None")

    # Print and save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(config["num_classes"]))
    # disp.plot(cmap=plt.cm.Blues)
    # plt.title("Confusion Matrix")
    # plt.savefig(config["conf_matrix_path"])
    # plt.close()

if __name__ == "__main__":
    train_model(config)
