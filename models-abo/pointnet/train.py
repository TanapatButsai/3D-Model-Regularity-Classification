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

# Configuration
config = {
    "label_file": "datasets/abo/label/Final_Validated_Regularity_Levels.xlsx",
    "obj_folder": "datasets/abo/obj-ABO",
    "num_points": 1024,  # Number of points to sample from each 3D object
    "batch_size": 16,
    "num_epochs": 60,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "dropout_rate": 0.3,
    "class_weights": [1.0, 2.0, 3.0, 4.0],  # Adjust these based on class distribution
    "test_size": 0.2,
    "num_of_sample": 7583,  # Number of samples to use for training/testing
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

print(f"Using device: {config['device']}")

# Load labels from Excel file
label_data = pd.read_excel(config["label_file"])

# Limit the number of samples if specified in the config
if config["num_of_sample"] < len(label_data):
    label_data = label_data.sample(n=config["num_of_sample"], random_state=42)

# Extract relevant columns
labels = label_data[['Object ID (Dataset Original Object ID)', 'Final Regularity Level']]

# Helper function to load point cloud from OBJ file
def load_pointcloud_from_obj(file_path, num_points=config["num_points"]):
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
    obj_file = os.path.join(config["obj_folder"], object_id.strip(), f'{object_id.strip()}.obj')
    
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
    torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long),
    test_size=config["test_size"], random_state=42
)

# Move data to device
X_train, X_test = X_train.to(config["device"]), X_test.to(config["device"])
y_train, y_test = y_train.to(config["device"]), y_test.to(config["device"])

# Define the PointNet model with batch normalization and increased capacity
class ImprovedPointNet(nn.Module):
    def __init__(self, k=num_classes):
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
        self.dropout = nn.Dropout(p=config["dropout_rate"])

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
model = ImprovedPointNet(k=num_classes).to(config["device"])

# Define optimizer and weighted loss function to handle class imbalance
class_weights = torch.tensor(config["class_weights"], device=config["device"])
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Training loop
train_data = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)

test_data = torch.utils.data.TensorDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=config["batch_size"], shuffle=False)

for epoch in range(config["num_epochs"]):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.transpose(2, 1), labels
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{config['num_epochs']}], Loss: {running_loss/len(train_loader):.4f}")

# Evaluation and metrics
# Path to save the results
results_path = "datasets/abo/label/pointnet_evaluation_results.csv"

# Evaluation and metrics
model.eval()
y_true = []
y_pred = []
y_pred_prob = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.transpose(2, 1), labels
        outputs = model(inputs)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
        y_pred_prob.extend(probabilities.cpu().numpy())

# Confusion Matrix and Metrics
conf_matrix = confusion_matrix(y_true, y_pred)
accuracy = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)  # Accuracy calculation
precision = precision_score(y_true, y_pred, average="weighted")
recall = recall_score(y_true, y_pred, average="weighted")
f1 = f1_score(y_true, y_pred, average="weighted")
y_true_one_hot = np.eye(num_classes)[y_true]
auc_roc = roc_auc_score(y_true_one_hot, y_pred_prob, multi_class="ovr")
log_loss_value = log_loss(y_true, y_pred_prob)

# Print results
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
print(f"AUC-ROC: {auc_roc:.4f}, Log Loss: {log_loss_value:.4f}")

# Save results to a CSV file
results = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC", "Log Loss"],
    "Value": [accuracy, precision, recall, f1, auc_roc, log_loss_value]
}
results_df = pd.DataFrame(results)

# Save or append to CSV
if not os.path.exists(results_path):
    results_df.to_csv(results_path, index=False)
else:
    results_df.to_csv(results_path, mode='a', header=False, index=False)

print(f"Evaluation results saved to {results_path}")

# Save confusion matrix as an image
conf_matrix_path = os.path.join("datasets", "abo", "label", "PointNet_matrix.png")

plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(np.unique(y_true)))
plt.xticks(tick_marks, [f"Class_{i}" for i in range(len(np.unique(y_true)))], rotation=45)
plt.yticks(tick_marks, [f"Class_{i}" for i in range(len(np.unique(y_true)))])

# Add values inside the confusion matrix
thresh = conf_matrix.max() / 2
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(
            j, i, format(conf_matrix[i, j], "d"),
            horizontalalignment="center",
            color="white" if conf_matrix[i, j] > thresh else "black"
        )

plt.tight_layout()
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.savefig(conf_matrix_path)
plt.close()

print(f"Confusion matrix saved as {conf_matrix_path}")
