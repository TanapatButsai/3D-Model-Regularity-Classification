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

# Configuration dictionary
config = {
    "num_points": 1024,
    "batch_size": 32,
    "num_classes": 4,
    "num_epochs": 70,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "dataset_folder": 'datasets/IKEA/obj-IKEA',
    "label_file_path": 'datasets/IKEA/label/Final_Validated_Regularity_Levels.xlsx',
    "patience": 5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

def print_config(config):
    print("Configuration Settings:")
    print("-----------------------")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("-----------------------\n")

print_config(config)

# Extract mesh features (vertices, faces, normals)
def extract_mesh_features(file_path, max_vertices=1024):
    try:
        mesh = trimesh.load(file_path)
        if isinstance(mesh, trimesh.Scene):
            if mesh.geometry:
                mesh = trimesh.util.concatenate(mesh.geometry.values())
            else:
                print(f"Scene has no geometries: {file_path}")
                return None, None, None
        if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces'):
            print(f"Mesh is missing vertices or faces: {file_path}")
            return None, None, None

        vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int32)
        normals = np.array(mesh.vertex_normals, dtype=np.float32)

        # Pad or truncate to max_vertices
        if vertices.shape[0] > max_vertices:
            indices = np.random.choice(vertices.shape[0], max_vertices, replace=False)
            vertices = vertices[indices]
            normals = normals[indices]
        elif vertices.shape[0] < max_vertices:
            padding = np.zeros((max_vertices - vertices.shape[0], 3), dtype=np.float32)
            vertices = np.vstack([vertices, padding])
            normals = np.vstack([normals, padding])
        
        return vertices, faces, normals
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None, None

# Dataset class for IKEA
class MeshDataset(Dataset):
    def __init__(self, labels_df, dataset_folder, max_vertices=1024, augment=False):
        self.labels_df = labels_df
        self.dataset_folder = dataset_folder
        self.max_vertices = max_vertices
        self.augment = augment

    def __len__(self):
        return len(self.labels_df)

    def augment_vertices(self, vertices):
        # Apply random rotation
        theta = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                     [np.sin(theta), np.cos(theta), 0],
                                     [0, 0, 1]])
        vertices = np.dot(vertices, rotation_matrix)

        # Apply random scaling
        scale = np.random.uniform(0.9, 1.1)
        vertices *= scale

        # Apply random jittering
        jitter = np.random.normal(0, 0.02, size=vertices.shape)
        vertices += jitter

        return vertices

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        obj_id = row['Object ID (Dataset Original Object ID)']
        folder_name = row['FolderName']
        label = int(row['Final Regularity Level']) - 1
        obj_file = os.path.join(self.dataset_folder, folder_name.strip(), obj_id.strip(), 'ikea_model.obj')

        vertices, faces, normals = extract_mesh_features(obj_file, self.max_vertices)
        if vertices is None or normals is None:
            vertices = np.zeros((self.max_vertices, 3), dtype=np.float32)
            normals = np.zeros((self.max_vertices, 3), dtype=np.float32)
        else:
            if self.augment:
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
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, vertices, normals):
        # Concatenate vertices and normals along the channel dimension
        x = torch.cat([vertices, normals], dim=2)
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = torch.max(x, dim=-1)[0]
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# Training function
def train_meshnet(config):
    # Load dataset and split into train/test
    labels_df = pd.read_excel(config["label_file_path"])
    train_df, test_df = train_test_split(labels_df, test_size=0.2, stratify=labels_df['Final Regularity Level'], random_state=42)

    print("Training set class distribution:")
    print(train_df['Final Regularity Level'].value_counts())
    print("Testing set class distribution:")
    print(test_df['Final Regularity Level'].value_counts())

    train_dataset = MeshDataset(train_df, config["dataset_folder"], augment=True)
    test_dataset = MeshDataset(test_df, config["dataset_folder"], augment=False)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # Model setup
    model = MeshNet(config["num_classes"]).to(config["device"])
    
    # Class weights
    class_counts = train_df['Final Regularity Level'].value_counts()
    class_weights = 1.0 / class_counts
    weights = torch.tensor([class_weights[i] for i in range(1, config["num_classes"] + 1)], dtype=torch.float32).to(config["device"])
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Training loop
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
        scheduler.step()

    # Evaluation
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

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
    conf_matrix = confusion_matrix(y_true, y_pred)

    if len(np.unique(y_true)) > 1:  # Ensure at least two classes are present
        auc_roc = roc_auc_score(
            np.eye(config["num_classes"])[y_true], y_prob, multi_class='ovr'
        )
        print(f"AUC-ROC: {auc_roc:.2f}")
    else:
        auc_roc = None
        print("AUC-ROC: Not defined (only one class present in y_true).")


    logloss = log_loss(y_true, y_prob)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    if auc_roc is not None:
        print(f"AUC-ROC: {auc_roc:.2f}")
    else:
        print("AUC-ROC: Not defined (only one class present in y_true).")
    print(f"Log Loss: {logloss:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")

train_meshnet(config)
