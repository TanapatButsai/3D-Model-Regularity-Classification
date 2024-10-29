import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import trimesh
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.utils import class_weight

# Load labels from Excel file
label_file = 'datasets/ShapeNetCoreV2/label/Final_Validated_Regularity_Levels.xlsx'
label_data = pd.read_excel(label_file)

# Extract relevant columns
labels = label_data[['Object ID (Dataset Original Object ID)', 'Final Regularity Level', 'Folder Name']]

# Path to the folder containing 3D objects
obj_folder = 'datasets/ShapeNetCoreV2/obj-ShapeNetCoreV2'

# PointNet model definition (with increased dropout to combat overfitting)
class PointNet(nn.Module):
    def __init__(self, k=4):
        super(PointNet, self).__init__()
        
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.conv3 = nn.Conv1d(256, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.6)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # Dynamic max pooling based on input size
        pool_size = x.size(2)  # Use the actual number of points in the input
        x = nn.functional.max_pool1d(x, kernel_size=pool_size)
        
        x = x.view(-1, 1024)  # Reshape for fully connected layers
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x



# Function to load point cloud from an OBJ file and return it as a numpy array
def load_pointcloud_from_obj(file_path, num_points=1024):
    try:
        mesh = trimesh.load(file_path)
        points = mesh.sample(num_points)
        return points
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Apply random jittering, scaling, rotation, and point dropout as data augmentation
def augment_pointcloud(points, dropout_ratio=0.05):
    # Apply random jittering (adding small noise)
    jitter = np.random.normal(0, 0.02, size=points.shape)
    points += jitter

    # Apply random scaling
    scale = np.random.uniform(0.8, 1.2)
    points *= scale

    # Apply random rotation around the z-axis
    theta = np.random.uniform(0, 2 * np.pi)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    points = np.dot(points, rotation_matrix.T)

    # Apply random point dropout
    num_points = points.shape[0]
    num_drop = int(dropout_ratio * num_points)
    keep_idx = np.random.choice(num_points, num_points - num_drop, replace=False)
    points = points[keep_idx, :]
    
    return points

# Prepare dataset for PointNet
point_clouds = []
targets = []

for index, row in tqdm(labels.iterrows(), total=len(labels)):
    first_layer_folder = str(int(row['Folder Name'])).zfill(8)
    second_layer_folder = str(row['Object ID (Dataset Original Object ID)']).strip()
    obj_filename = 'model_normalized'
    obj_file = os.path.join(obj_folder, first_layer_folder, second_layer_folder, 'models', f"{obj_filename}.obj")
    
    # Load point cloud from the OBJ file and apply augmentation
    if os.path.isfile(obj_file):
        point_cloud = load_pointcloud_from_obj(obj_file)
        if point_cloud is not None:
            point_cloud = augment_pointcloud(point_cloud)
            point_clouds.append(point_cloud)
            targets.append(row['Final Regularity Level'])

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

# Correct the class weight computation
class_weights_np = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weights = torch.tensor(class_weights_np, dtype=torch.float)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Initialize the PointNet model, optimizer, loss function with class weighting, and learning rate scheduler
model = PointNet(k=len(np.unique(y_train)))  # Set the number of classes dynamically

criterion = nn.CrossEntropyLoss(weight=class_weights)

# Switch optimizer to SGD with momentum
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)  # Learning rate scheduler

# Training loop for PointNet
num_epochs = 50
batch_size = 8  # Adjusted batch size

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
    
    scheduler.step()  # Update learning rate
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
