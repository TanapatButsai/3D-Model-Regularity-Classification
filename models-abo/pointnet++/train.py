import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score,
    recall_score, f1_score, roc_auc_score, log_loss
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import trimesh
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration
config = {
    "label_file": "datasets/abo/label/Final_Validated_Regularity_Levels.xlsx",
    "obj_folder": "datasets/abo/obj-ABO",
    "num_points": 1024,
    "batch_size": 32,
    "num_epochs": 80,
    "learning_rate": 0.0005,
    "weight_decay": 1e-5,
    "dropout_rate": 0.4,
    "test_size": 0.2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "results_folder": "datasets/abo/label"
}

os.makedirs(config["results_folder"], exist_ok=True)

# Load point clouds from OBJ files
def load_pointcloud_from_obj(file_path, num_points):
    try:
        mesh = trimesh.load(file_path, force='mesh')
        return mesh.sample(num_points)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Augment data with rotations and noise
def augment_pointcloud(points):
    rotation_matrix = trimesh.transformations.random_rotation_matrix()[:3, :3]
    points = np.dot(points, rotation_matrix.T)
    points += np.random.normal(0, 0.02, points.shape)
    scale_factor = np.random.uniform(0.8, 1.2)
    points *= scale_factor
    return points

# Data Preprocessing
def preprocess_data(config):
    labels_df = pd.read_excel(config["label_file"])
    point_clouds, labels = [], []

    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
        object_id = row["Object ID (Dataset Original Object ID)"]
        label = row["Final Regularity Level"]
        obj_file = os.path.join(config["obj_folder"], object_id.strip(), f"{object_id.strip()}.obj")
        
        if os.path.isfile(obj_file):
            points = load_pointcloud_from_obj(obj_file, config["num_points"])
            if points is not None:
                point_clouds.append(augment_pointcloud(points))
                labels.append(label)

    X = np.array(point_clouds)
    y = np.array(labels)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    return train_test_split(
        torch.tensor(X, dtype=torch.float32).permute(0, 2, 1),
        torch.tensor(y, dtype=torch.long),
        test_size=config["test_size"], random_state=42
    )

# PointNet++ Set Abstraction Layer
class SetAbstraction(nn.Module):
    def __init__(self, in_channels, out_channels, num_points):
        super(SetAbstraction, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels // 2, 1)
        self.conv2 = nn.Conv1d(out_channels // 2, out_channels, 1)
        self.bn1 = nn.BatchNorm1d(out_channels // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.num_points = num_points

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = torch.max(x, dim=-1)[0]
        return x.unsqueeze(-1)

# PointNet++ Model
class PointNetPlusPlus(nn.Module):
    def __init__(self, num_classes, num_points, dropout_rate):
        super(PointNetPlusPlus, self).__init__()
        self.sa1 = SetAbstraction(3, 64, num_points // 2)
        self.sa2 = SetAbstraction(64, 128, num_points // 4)
        self.sa3 = SetAbstraction(128, 256, num_points // 8)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.sa1(x)
        x = self.sa2(x)
        x = self.sa3(x)
        x = x.squeeze(-1)  # Flatten features
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# Train the Model
def train_model():
    X_train, X_test, y_train, y_test = preprocess_data(config)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train),
                                               batch_size=config["batch_size"], shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, y_test),
                                              batch_size=config["batch_size"], shuffle=False)

    model = PointNetPlusPlus(num_classes=4, num_points=config["num_points"], dropout_rate=config["dropout_rate"]).to(config["device"])
    class_weights = 1.0 / torch.bincount(y_train)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(config["device"]))
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(config["device"]), y_batch.to(config["device"])
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{config['num_epochs']}], Loss: {total_loss/len(train_loader):.4f}")

    evaluate_model(model, test_loader, y_test)

# Evaluate the Model
def evaluate_model(model, test_loader, y_test):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(config["device"]), y_batch.to(config["device"])
            outputs = model(X_batch)
            prob = torch.softmax(outputs, dim=1)
            pred = torch.argmax(prob, dim=1)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_prob.extend(prob.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    auc = roc_auc_score(pd.get_dummies(y_true), y_prob, average="weighted", multi_class="ovr")
    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}")

    # Save Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.savefig(os.path.join(config["results_folder"], "confusion_matrix.png"))
    plt.close()

    # Save Metrics
    metrics = {"Accuracy": acc, "Precision": precision, "Recall": recall, "F1 Score": f1, "AUC": auc}
    pd.DataFrame([metrics]).to_csv(os.path.join(config["results_folder"], "metrics.csv"), index=False)

# Run Training and Evaluation
if __name__ == "__main__":
    train_model()
