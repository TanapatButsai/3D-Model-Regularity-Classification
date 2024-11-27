import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score,
    f1_score, roc_auc_score
)
import trimesh
from tqdm import tqdm

# Configuration
config = {
    "label_file": "datasets/pix3d/label/Final_Validated_Regularity_Levels.xlsx",
    "obj_folder": "datasets/pix3d/obj-pix3d",
    "num_points": 1024,
    "batch_size": 16,
    "num_epochs": 50,
    "learning_rate": 0.001,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# Load labels from Excel
label_data = pd.read_excel(config["label_file"])
labels = label_data[['Object ID (Dataset Original Object ID)', 'Final Regularity Level']]

# Function to load point clouds
def load_pointcloud(file_path, num_points):
    try:
        mesh = trimesh.load(file_path)
        if isinstance(mesh, trimesh.Scene) and len(mesh.geometry) > 0:
            mesh = list(mesh.geometry.values())[0]
        if not hasattr(mesh, 'vertices'):
            return None
        return mesh.sample(num_points)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Prepare dataset
point_clouds = []
targets = []

for _, row in tqdm(labels.iterrows(), total=len(labels)):
    object_id = row['Object ID (Dataset Original Object ID)']
    regularity_level = row['Final Regularity Level']
    obj_path = os.path.join(config["obj_folder"], "models", object_id.strip(), "model.obj")
    if os.path.isfile(obj_path):
        point_cloud = load_pointcloud(obj_path, config["num_points"])
        if point_cloud is not None:
            point_clouds.append(point_cloud)
            targets.append(regularity_level)

if not point_clouds:
    print("No point clouds loaded. Check the dataset.")
    exit()

X = np.array(point_clouds)
y = np.array(targets)

# Filter out classes with fewer than 2 samples
class_counts = np.bincount(y)
valid_classes = np.where(class_counts >= 2)[0]
valid_indices = np.isin(y, valid_classes)
X = X[valid_indices]
y = y[valid_indices]

# Re-encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
num_classes = len(np.unique(y))

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train, y_train),
    batch_size=config["batch_size"], shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_test, y_test),
    batch_size=config["batch_size"], shuffle=False
)

# PointNet Model
class PointNet(nn.Module):
    def __init__(self, num_classes):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        return x

# Initialize model, loss, and optimizer
model = PointNet(num_classes).to(config["device"])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# Training loop
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
    print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {running_loss/len(train_loader):.4f}")

# Evaluation
model.eval()
y_true, y_pred, y_prob = [], [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(config["device"]), labels.to(config["device"])
        outputs = model(inputs)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
        y_prob.extend(probabilities.cpu().numpy())

# Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
conf_matrix = confusion_matrix(y_true, y_pred)

# AUC-ROC Calculation
y_true_one_hot = np.zeros((len(y_true), num_classes))
y_true_one_hot[np.arange(len(y_true)), y_true] = 1
try:
    auc_roc = roc_auc_score(y_true_one_hot, np.array(y_prob), multi_class='ovr', average='weighted')
except ValueError as e:
    auc_roc = f"N/A ({str(e)})"

# Print Metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"AUC-ROC: {auc_roc}")
print(f"Confusion Matrix:\n{conf_matrix}")
