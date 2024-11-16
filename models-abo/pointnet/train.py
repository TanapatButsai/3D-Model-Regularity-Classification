import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import trimesh
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss
)
import matplotlib.pyplot as plt

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load labels from Excel file
label_file = 'datasets/abo/label/Final_Validated_Regularity_Levels.xlsx'
label_data = pd.read_excel(label_file)

# Define MAX_DATA_POINTS
MAX = len(label_data)
MAX_DATA_POINTS = MAX  # You can change this number based on how many data points you want to train with

# Limit the number of data points
if len(label_data) > MAX_DATA_POINTS:
    label_data = label_data.sample(n=MAX_DATA_POINTS, random_state=42)

# Extract relevant columns
labels = label_data[['Object ID (Dataset Original Object ID)', 'Final Regularity Level']]

# Path to the folder containing 3D objects
obj_folder = 'datasets/abo/obj-ABO'

# Helper function to load point cloud from OBJ file
def load_pointcloud_from_obj(file_path, num_points=1024):
    try:
        mesh = trimesh.load(file_path)
        points = mesh.sample(num_points)
        return points
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Prepare dataset for PointNet
point_clouds = []
targets = []

for index, row in tqdm(labels.iterrows(), total=len(labels)):
    object_id = row['Object ID (Dataset Original Object ID)']
    regularity_level = row['Final Regularity Level']
    
    # Construct the path
    obj_file = os.path.join(obj_folder, object_id.strip(), f'{object_id.strip()}.obj')
    
    # Load point cloud
    if os.path.isfile(obj_file):
        point_cloud = load_pointcloud_from_obj(obj_file)
        if point_cloud is not None:
            point_clouds.append(point_cloud)
            targets.append(regularity_level)

# Convert to numpy arrays
if len(point_clouds) == 0:
    print("No point clouds loaded. Please check the dataset and file paths.")
    exit()

X = np.array(point_clouds)
y = np.array(targets)

# Label encoding for continuous classes starting from 0
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Count the number of unique labels (classes)
num_classes = len(np.unique(y))
print(f"Number of unique classes: {num_classes}")

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long), test_size=0.2, random_state=42
)

# Move data to device
X_train, X_test = X_train.to(device), X_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)

# Define the PointNet model with batch normalization and increased capacity
class ImprovedPointNet(nn.Module):
    def __init__(self, k=4):
        super(ImprovedPointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, k)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(1024)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x).view(-1, 1024)
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Initialize PointNet model with the correct number of classes
model = ImprovedPointNet(k=num_classes).to(device)

# Define optimizer and weighted loss function to handle class imbalance
class_weights = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)  # Adjust weights based on class distribution
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Training loop
num_epochs = 30
batch_size = 16

train_data = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_data = torch.utils.data.TensorDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device).transpose(2, 1), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Evaluation and metrics
model.eval()
y_true = []
y_pred = []
y_pred_prob = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device).transpose(2, 1), labels.to(device)
        outputs = model(inputs)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
        y_pred_prob.extend(probabilities.cpu().numpy())

# Convert lists to NumPy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_pred_prob = np.array(y_pred_prob)

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Precision, Recall, F1 Score
precision = precision_score(y_true, y_pred, average="weighted")
recall = recall_score(y_true, y_pred, average="weighted")
f1 = f1_score(y_true, y_pred, average="weighted")
print(f"Precision: {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# AUC-ROC
y_true_one_hot = np.eye(num_classes)[y_true]  # One-hot encoding for AUC-ROC
auc_roc = roc_auc_score(y_true_one_hot, y_pred_prob, multi_class="ovr")
print(f"AUC-ROC: {auc_roc:.4f}")

# Log Loss
log_loss_value = log_loss(y_true, y_pred_prob)
print(f"Log Loss: {log_loss_value:.4f}")
