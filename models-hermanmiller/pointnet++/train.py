import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score,
    recall_score, f1_score, roc_auc_score, classification_report
)
from sklearn.model_selection import train_test_split
import trimesh
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelBinarizer

# Configuration
config = {
    "dataset_folder": "datasets/hermanmiller/obj-hermanmiller",
    "label_file": "datasets/hermanmiller/label/Final_Validated_Regularity_Levels.xlsx",
    "num_points": 1024,
    "batch_size": 32,
    "num_epochs": 100,
    "learning_rate": 0.0005,
    "weight_decay": 1e-5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_classes": 4,
    "conf_matrix_path": "datasets/hermanmiller/conf_matrix.png"
}

# Load 3D point cloud from OBJ file
def load_pointcloud(obj_path, num_points):
    try:
        mesh = trimesh.load(obj_path, force='mesh')
        if isinstance(mesh, trimesh.Trimesh):
            return mesh.sample(num_points)
    except Exception as e:
        print(f"Error loading {obj_path}: {e}")
    return None

# Data augmentation
def augment_pointcloud(points):
    rotation_matrix = trimesh.transformations.random_rotation_matrix()[:3, :3]
    points = np.dot(points, rotation_matrix.T)
    points += np.random.normal(0, 0.02, points.shape)
    return points

# Data preprocessing
def preprocess_data(config):
    labels_df = pd.read_excel(config["label_file"])
    point_clouds, labels = [], []
    
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
        object_id = row['Object ID (Dataset Original Object ID)']
        label = row['Final Regularity Level'] - 1
        obj_path = os.path.join(config["dataset_folder"], object_id.strip(), f"{object_id.strip()}.obj")
        
        if os.path.isfile(obj_path):
            point_cloud = load_pointcloud(obj_path, config["num_points"])
            if point_cloud is not None:
                point_clouds.append(augment_pointcloud(point_cloud))
                labels.append(label)

    point_clouds, labels = np.array(point_clouds), np.array(labels)
    
    # Handle class imbalance using SMOTE
    smote = SMOTE()
    point_clouds_flat = point_clouds.reshape(point_clouds.shape[0], -1)
    point_clouds_resampled, labels_resampled = smote.fit_resample(point_clouds_flat, labels)
    point_clouds_resampled = point_clouds_resampled.reshape(-1, config["num_points"], 3)
    
    return train_test_split(
        torch.tensor(point_clouds_resampled, dtype=torch.float32).permute(0, 2, 1),
        torch.tensor(labels_resampled, dtype=torch.long),
        test_size=0.2,
        random_state=42
    )

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
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.bn_fc2 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]  # Global Max Pooling
        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn_fc2(self.fc2(x)))
        x = self.fc3(x)
        return x

# Train the model
def train_model(config):
    X_train, X_test, y_train, y_test = preprocess_data(config)
    train_loader = DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(torch.utils.data.TensorDataset(X_test, y_test), batch_size=config["batch_size"], shuffle=False)
    
    model = PointNetPlusPlus(config["num_classes"]).to(config["device"])
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    for epoch in range(config["num_epochs"]):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(config["device"]), labels.to(config["device"])
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {running_loss/len(train_loader):.4f}")

    evaluate_model(model, test_loader, config)

# Evaluate the model
def evaluate_model(model, test_loader, config):
    model.eval()
    all_preds, all_labels, all_probas = [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(config["device"]), labels.to(config["device"])
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probas.extend(probs.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=1)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=1)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=1)
    auc_roc = roc_auc_score(LabelBinarizer().fit_transform(all_labels), all_probas, average="weighted", multi_class="ovr")

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"AUC-ROC: {auc_roc:.2f}")

    # Save confusion matrix as image
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(config["conf_matrix_path"])
    plt.close()

if __name__ == "__main__":
    train_model(config)
