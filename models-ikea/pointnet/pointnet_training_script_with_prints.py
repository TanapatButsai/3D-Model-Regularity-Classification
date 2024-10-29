import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import trimesh
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Function to load point cloud from an OBJ file and return it as a numpy array
def load_pointcloud_from_obj(file_path, num_points=1024):
    try:
        mesh = trimesh.load(file_path)
        points = mesh.sample(num_points)
        return points
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Function to process the dataset structure, assuming each folder contains 'ikea_model.obj'
def process_dataset_structure(base_folder, labels):
    point_clouds = []
    targets = []

    # Iterate through the labels to construct file paths and load data
    for index, row in tqdm(labels.iterrows(), total=len(labels)):
        object_id = row['Object ID (Dataset Original Object ID)']
        regularity_level = row['Final Regularity Level']
        folder_name = row['FolderName']

        # Construct the path to the obj file using FolderName and Object ID
        obj_folder = os.path.join(base_folder, folder_name.strip(), object_id.strip())
        obj_file = os.path.join(obj_folder, 'ikea_model.obj')

        # Load the point cloud from the obj file
        if os.path.isfile(obj_file):
            point_cloud = load_pointcloud_from_obj(obj_file)
            if point_cloud is not None:
                point_clouds.append(point_cloud)
                targets.append(regularity_level)
        else:
            print(f"OBJ file not found: {obj_file}")

    return np.array(point_clouds), np.array(targets)

# Path to your base folder and Excel file
base_folder = 'datasets\ikea\obj-IKEA'  # Replace with the path to your dataset folder
label_file = 'datasets\ikea\label\Final_Validated_Regularity_Levels.xlsx'  # Replace with the actual Excel file path

# Load labels from the Excel file
label_data = pd.read_excel(label_file)

# Process the dataset to extract point clouds and corresponding labels
point_clouds, targets = process_dataset_structure(base_folder, label_data)

# Check if point clouds were loaded
if len(point_clouds) == 0:
    print("No point clouds loaded. Please check the dataset and file paths.")
    exit()

# Convert to numpy arrays
X = np.array(point_clouds)
y = np.array(targets) - np.min(targets)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# PointNet model definition
class PointNet(nn.Module):
    def __init__(self, k=4):
        super(PointNet, self).__init__()
        
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(1024)
        self.dropout = nn.Dropout(p=0.3)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # Max pooling to obtain a global feature vector
        x = self.maxpool(x)
        x = x.view(-1, 1024)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# Initialize the PointNet model, loss function, and optimizer
model = PointNet(k=4)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop for PointNet
num_epochs = 10
batch_size = 4

train_data = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_data = torch.utils.data.TensorDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        # Transpose inputs to match the shape expected by PointNet (batch_size, num_points, 3)
        inputs = inputs.transpose(2, 1)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Evaluation on the test set
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.transpose(2, 1)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
