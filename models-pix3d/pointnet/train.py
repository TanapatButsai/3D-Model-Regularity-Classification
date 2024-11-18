import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import trimesh
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score, roc_auc_score, log_loss)

# Configuration dictionary
CONFIG = {
    "dataset_folder": "datasets/pix3d/obj-pix3d/models",
    "excel_path": "datasets/pix3d/label/Final_Validated_Regularity_Levels.xlsx",
    "num_epochs": 30,
    "batch_size": 4,
    "num_points": 1024,
    "learning_rate": 0.001,
    "step_size": 10,
    "gamma": 0.5,
    "class_weights": [3.0, 0.8, 2.0, 1.0],
    "device": "cpu"
}

# PointNet Model Definition
class PointNet(nn.Module):
    def __init__(self, k=4):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.conv3 = nn.Conv1d(256, 512, 1)
        self.conv4 = nn.Conv1d(512, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(1024)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.maxpool(x)
        x = x.view(-1, 1024)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Convert OBJ file to point cloud
def obj_to_pointcloud(obj_path, num_points=1024):
    try:
        mesh = trimesh.load(obj_path)
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) > 0:
                mesh = list(mesh.geometry.values())[0]
            else:
                return None
        if not isinstance(mesh, trimesh.Trimesh):
            return None
        points = mesh.sample(num_points)
        return points
    except Exception:
        return None

# Apply advanced data augmentation
def augment_pointcloud(points):
    jitter = np.random.normal(0, 0.02, size=points.shape)
    points += jitter
    theta = np.random.uniform(0, 2*np.pi)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta),  np.cos(theta), 0],
                                [0,              0,             1]])
    points = np.dot(points, rotation_matrix.T)
    scale = np.random.uniform(0.8, 1.2)
    points *= scale
    if np.random.rand() > 0.5:
        points[:, 0] = -points[:, 0]
    return points

# Process the dataset to convert OBJ files into point clouds
def process_dataset(config):
    print("Starting dataset extraction and point cloud conversion...")
    point_clouds = []
    labels = []
    excel_data = pd.read_excel(config["excel_path"])

    for _, row in tqdm(excel_data.iterrows(), total=len(excel_data), desc="Processing OBJ Files"):
        object_id = row['Object ID (Dataset Original Object ID)']
        label = row['Final Regularity Level'] - 1

        obj_folder = os.path.join(config["dataset_folder"], object_id)
        obj_file = os.path.join(obj_folder, "model.obj")

        if os.path.exists(obj_file):
            point_cloud = obj_to_pointcloud(obj_file, num_points=config["num_points"])
            if point_cloud is not None:
                point_cloud = augment_pointcloud(point_cloud)
                point_clouds.append(point_cloud)
                labels.append(label)

    print("Finished dataset processing.")
    return np.array(point_clouds), np.array(labels)

# Main training loop
def train_pointnet(config):
    point_clouds, labels = process_dataset(config)

    point_clouds_tensor = torch.tensor(point_clouds, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    class PointCloudDataset(torch.utils.data.Dataset):
        def __init__(self, point_clouds, labels):
            self.point_clouds = point_clouds
            self.labels = labels
        
        def __len__(self):
            return len(self.point_clouds)
        
        def __getitem__(self, idx):
            return self.point_clouds[idx], self.labels[idx]

    dataset = PointCloudDataset(point_clouds_tensor, labels_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    class_weights = torch.tensor(config["class_weights"], dtype=torch.float32).to(config["device"])
    model = PointNet(k=len(config["class_weights"])).to(config["device"])
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])

    for epoch in range(config["num_epochs"]):
        epoch_loss = 0
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{config['num_epochs']}") as pbar:
            for inputs, target in dataloader:
                inputs = inputs.transpose(2, 1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
                pbar.update(1)
        scheduler.step()
        print(f"Epoch [{epoch+1}/{config['num_epochs']}], Average Loss: {epoch_loss/len(dataloader):.4f}")

    print("Training completed!")
    
    # Evaluation
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating Model"):
            inputs = inputs.transpose(2, 1)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=1)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)

    try:
        one_hot_labels = np.zeros((all_labels.size, all_probs.shape[1]))
        one_hot_labels[np.arange(all_labels.size), all_labels] = 1
        auc_roc = roc_auc_score(one_hot_labels, all_probs, multi_class='ovr', average="weighted")
    except ValueError:
        auc_roc = "N/A"

    try:
        logloss = log_loss(all_labels, all_probs)
    except ValueError:
        logloss = "N/A"

    print("\nEvaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc}")
    print(f"Log Loss: {logloss}")

# Example usage
if __name__ == "__main__":
    train_pointnet(CONFIG)
