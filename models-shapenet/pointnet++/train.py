import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from tqdm import tqdm
import trimesh
import random
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, 
                             recall_score, f1_score, roc_auc_score, log_loss)

# Configuration dictionary
config = {
    "num_samples": 7500,
    "num_epochs": 50,
    "batch_size": 32,
    "num_points": 1024,
    "learning_rate": 0.001,
    "dataset_folder": 'datasets/ShapeNetCoreV2/obj-ShapeNetCoreV2',
    "excel_path": 'datasets/ShapeNetCoreV2/label/final_regularized_labels.xlsx',
    "patience": 5,  # For early stopping
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

def print_config(config):
    print("Configuration Settings:")
    print("-----------------------")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("-----------------------\n")

print_config(config)

def fix_point_cloud_size(point_cloud, num_points=1024):
    if point_cloud.shape[0] > num_points:
        indices = np.random.choice(point_cloud.shape[0], num_points, replace=False)
        point_cloud = point_cloud[indices, :]
    elif point_cloud.shape[0] < num_points:
        indices = np.random.choice(point_cloud.shape[0], num_points - point_cloud.shape[0], replace=True)
        point_cloud = np.concatenate([point_cloud, point_cloud[indices, :]], axis=0)
    return point_cloud

def augment_point_cloud(point_cloud):
    theta = random.uniform(0, 2 * np.pi)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    point_cloud = np.dot(point_cloud, rotation_matrix)
    scale = random.uniform(0.9, 1.1)
    point_cloud *= scale
    return point_cloud

def extract_features_from_obj(file_path, num_points=1024):
    try:
        loaded_obj = trimesh.load(file_path)
        if isinstance(loaded_obj, trimesh.Scene):
            if loaded_obj.geometry:
                mesh = trimesh.util.concatenate(loaded_obj.geometry.values())
            else:
                print(f"Scene has no geometries: {file_path}")
                return None
        else:
            mesh = loaded_obj
        if not hasattr(mesh, 'vertices'):
            print(f"Mesh has no vertices: {file_path}")
            return None
        point_cloud = np.array(mesh.vertices)
        point_cloud = fix_point_cloud_size(point_cloud, num_points)
        point_cloud = augment_point_cloud(point_cloud)
        return point_cloud
    except ValueError as ve:
        print(f"ValueError loading {file_path}: {ve}")
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Dataset class for ShapeNet
class ShapeNetDataset(Dataset):
    def __init__(self, labels, obj_folder, num_points=1024):
        self.labels = labels
        self.obj_folder = obj_folder
        self.num_points = num_points
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        first_layer_folder = str(int(row['Folder Name'])).zfill(8)
        second_layer_folder = str(row['Object ID (Dataset Original Object ID)']).strip()
        obj_filename = 'model_normalized'
        obj_file = os.path.join(self.obj_folder, first_layer_folder, second_layer_folder, 'models', f"{obj_filename}.obj")
        point_cloud = extract_features_from_obj(obj_file, self.num_points)
        if point_cloud is None:
            point_cloud = np.zeros((self.num_points, 3))
        label = row['Final Regularity Level'] - 1
        return torch.tensor(point_cloud, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# PointNet++ model
class PointNetPlusPlus(nn.Module):
    def __init__(self, num_classes):
        super(PointNetPlusPlus, self).__init__()
        self.sa1 = nn.Conv1d(3, 64, 1)
        self.sa2 = nn.Conv1d(64, 128, 1)
        self.sa3 = nn.Conv1d(128, 256, 1)
        self.global_feat = nn.Conv1d(256, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.transpose(2, 1)
        x = self.relu(self.sa1(x))
        x = self.relu(self.sa2(x))
        x = self.relu(self.sa3(x))
        x = torch.max(self.global_feat(x), dim=-1, keepdim=False)[0]
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# Train PointNet++ model
def train_pointnet(config):
    label_data = pd.read_excel(config["excel_path"])
    labels = label_data[['Object ID (Dataset Original Object ID)', 'Final Regularity Level', 'Folder Name']]
    if config["num_samples"] < len(labels):
        labels = labels.sample(n=config["num_samples"], random_state=42)
    
    dataset = ShapeNetDataset(labels, config["dataset_folder"], num_points=config["num_points"])
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    num_classes = len(labels['Final Regularity Level'].unique())
    model = PointNetPlusPlus(num_classes).to(config["device"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_loss = float('inf')
    patience = 0

    for epoch in range(config["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}"):
            inputs, labels = inputs.to(config["device"]), labels.to(config["device"])
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{config['num_epochs']}], Loss: {avg_epoch_loss:.4f}")

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
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(config["device"]), labels.to(config["device"])
            outputs = model(inputs)
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
    print(f"Recall (Sensitivity): {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"AUC-ROC: {auc_roc:.2f}")
    print(f"Log Loss: {logloss:.4f}")

train_pointnet(config)
