import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, 
    recall_score, f1_score, roc_auc_score, log_loss, classification_report
)
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import trimesh
from tqdm import tqdm
import torch.nn.functional as F  # Import Functional API

# Configuration with adjustments
config = {
    "dataset_folder": "datasets/hermanmiller/obj-hermanmiller",
    "label_file": "datasets/hermanmiller/label/Final_Validated_Regularity_Levels.xlsx",
    "num_points": 1024,
    "batch_size": 16,
    "num_epochs": 100,     # Increased to 100 epochs
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "step_size": 20,
    "gamma": 0.5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_classes": 4,
    "class_weights": [1.5, 1.5, 1.8, 1.0]
}

print("Configuration Settings:")
for key, value in config.items():
    print(f"{key}: {value}")
print("\n")

# SetAbstraction with correct handling of input and output channels
class SetAbstraction(nn.Module):
    def __init__(self, in_channels, out_channels, num_points, sample_ratio):
        super(SetAbstraction, self).__init__()
        self.num_sampled_points = int(num_points * sample_ratio)
        self.conv1 = nn.Conv1d(in_channels, out_channels // 2, 1)
        self.conv2 = nn.Conv1d(out_channels // 2, out_channels, 1)
        self.bn1 = nn.BatchNorm1d(out_channels // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        if x.size(2) < self.num_sampled_points:
            raise ValueError(f"Not enough points to sample: required {self.num_sampled_points}, got {x.size(2)}")
        indices = torch.randperm(x.size(2))[:self.num_sampled_points]
        x = x[:, :, indices]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

# PointNet++ Model
class PointNetPlusPlus(nn.Module):
    def __init__(self, num_classes):
        super(PointNetPlusPlus, self).__init__()
        self.sa1 = SetAbstraction(3, 64, num_points=1024, sample_ratio=0.5)
        self.sa2 = SetAbstraction(64, 128, num_points=512, sample_ratio=0.25)
        self.fc1 = nn.Linear(128 * 128, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        x = self.sa1(x)
        x = self.sa2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

def obj_to_pointcloud(obj_path, num_points=config["num_points"]):
    mesh = trimesh.load(obj_path, force='mesh')
    if isinstance(mesh, trimesh.Trimesh):
        points = mesh.sample(num_points)
        return points
    return None

def process_dataset(dataset_folder, labels_df):
    point_clouds = []
    labels = []
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
        object_id = row['Object ID (Dataset Original Object ID)']
        label = row['Final Regularity Level'] - 1
        obj_file = os.path.join(dataset_folder, object_id.strip(), f"{object_id.strip()}.obj")
        if os.path.isfile(obj_file):
            point_cloud = obj_to_pointcloud(obj_file)
            if point_cloud is not None:
                point_clouds.append(point_cloud)
                labels.append(label)
    return np.array(point_clouds), np.array(labels)

def train_pointnetplusplus(config):
    labels_df = pd.read_excel(config["label_file"])
    point_clouds, labels = process_dataset(config["dataset_folder"], labels_df)
    X_train, X_test, y_train, y_test = train_test_split(
        point_clouds, labels, test_size=0.2, random_state=42
    )
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    device = config["device"]
    model = PointNetPlusPlus(num_classes=config["num_classes"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(config["class_weights"]).to(device))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    for epoch in range(config["num_epochs"]):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step(running_loss)
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {running_loss/len(train_loader):.4f}")

    model.eval()
    all_preds, all_labels, all_probas = [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probas.extend(probs.cpu().numpy())
    
    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=1)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=1)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=1)
    lb = LabelBinarizer()
    y_true_binarized = lb.fit_transform(all_labels)
    auc_roc = roc_auc_score(y_true_binarized, all_probas, average="weighted", multi_class="ovr")
    logloss = log_loss(all_labels, all_probas)

    # Print Metrics
    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:\n", conf_matrix)
    print(f"Precision: {precision:.2f}")
    print(f"Recall (Sensitivity): {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"AUC-ROC: {auc_roc:.2f}")
    print(f"Log Loss: {logloss:.2f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, zero_division=1))

if __name__ == "__main__":
    train_pointnetplusplus(config)
