import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from tqdm import tqdm
import trimesh
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, 
                             recall_score, f1_score, roc_auc_score, log_loss)

# Configuration dictionary
config = {
    "num_samples": 20000,
    "num_epochs": 70,
    "batch_size": 32,
    "learning_rate": 0.001,
    "dataset_folder": 'datasets/ShapeNetCoreV2/obj-ShapeNetCoreV2',
    "excel_path": 'datasets/ShapeNetCoreV2/label/final_regularized_labels.xlsx',
    "patience": 5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# config = {
#     "num_samples": 20000,
#     "num_epochs": 70,
#     "batch_size": 32,
#     "learning_rate": 0.001,
#     "dataset_folder": 'datasets/ShapeNetCoreV2/obj-ShapeNetCoreV2',
#     "excel_path": 'datasets/ShapeNetCoreV2/label/final_regularized_labels.xlsx',
#     "patience": 5,
#     "device": "cuda" if torch.cuda.is_available() else "cpu",
# }

def print_config(config):
    print("Configuration Settings:")
    print("-----------------------")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("-----------------------\n")

print_config(config)

def extract_mesh_features(file_path):
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
        return vertices, faces, normals
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None, None

class MeshDataset(Dataset):
    def __init__(self, labels, obj_folder, max_vertices=1024, augment=False):
        self.labels = labels
        self.obj_folder = obj_folder
        self.max_vertices = max_vertices
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def pad_or_truncate(self, array, target_size):
        if array.shape[0] > target_size:
            indices = np.random.choice(array.shape[0], target_size, replace=False)
            return array[indices]
        elif array.shape[0] < target_size:
            padding = np.zeros((target_size - array.shape[0], array.shape[1]), dtype=array.dtype)
            return np.concatenate([array, padding], axis=0)
        return array

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
        row = self.labels.iloc[idx]
        first_layer_folder = str(int(row['Folder Name'])).zfill(8)
        second_layer_folder = str(row['Object ID (Dataset Original Object ID)']).strip()
        obj_filename = 'model_normalized'
        obj_file = os.path.join(self.obj_folder, first_layer_folder, second_layer_folder, 'models', f"{obj_filename}.obj")
        vertices, faces, normals = extract_mesh_features(obj_file)
        if vertices is None or faces is None or normals is None:
            vertices = np.zeros((self.max_vertices, 3), dtype=np.float32)
            normals = np.zeros((self.max_vertices, 3), dtype=np.float32)
        else:
            vertices = self.pad_or_truncate(vertices, self.max_vertices)
            normals = self.pad_or_truncate(normals, self.max_vertices)
            if self.augment:
                vertices = self.augment_vertices(vertices)
        label = row['Final Regularity Level'] - 1
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
        x = torch.cat([vertices, normals], dim=2)  # Concatenate along the last dimension
        x = x.transpose(1, 2)  # Transpose to match input dimensions for Conv1D: [batch, channels, vertices]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = torch.max(x, dim=-1)[0]  # Global max pooling
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# Training function
def train_meshnet(config):
    label_data = pd.read_excel(config["excel_path"])
    labels = label_data[['Object ID (Dataset Original Object ID)', 'Final Regularity Level', 'Folder Name']]
    if config["num_samples"] < len(labels):
        labels = labels.sample(n=config["num_samples"], random_state=42)

    # Updated datasets with augmentation for training
    train_dataset = MeshDataset(labels, config["dataset_folder"], augment=True)
    test_dataset = MeshDataset(labels, config["dataset_folder"], augment=False)

    train_size = int(0.8 * len(train_dataset))
    test_size = len(train_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    num_classes = len(labels['Final Regularity Level'].unique())
    model = MeshNet(num_classes).to(config["device"])
    
    # Added weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
    patience = 0

    for epoch in range(config["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}"):
            inputs, labels = batch
            vertices, normals = inputs['vertices'].to(config["device"]), inputs['normals'].to(config["device"])
            labels = labels.to(config["device"])

            optimizer.zero_grad()
            outputs = model(vertices, normals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{config['num_epochs']}], Loss: {avg_epoch_loss:.4f}")

        # Early stopping
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience = 0
        else:
            patience += 1
            if patience >= config["patience"]:
                print("Early stopping triggered.")
                break

    # Evaluation loop
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            vertices, normals = inputs['vertices'].to(config["device"]), inputs['normals'].to(config["device"])
            labels = labels.to(config["device"])

            outputs = model(vertices, normals)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    auc_roc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    logloss = log_loss(all_labels, all_probs)

    print(f"Test Accuracy: {accuracy:.2f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"AUC-ROC: {auc_roc:.2f}")
    print(f"Log Loss: {logloss:.4f}")

train_meshnet(config)
