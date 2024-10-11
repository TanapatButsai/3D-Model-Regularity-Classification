import pandas as pd
import trimesh
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Configuration
base_dir = 'datasets/3d-future-dataset/3D-FUTURE-model'  # obj dir location
file_path = "datasets/3d-future-dataset/label/Final_Validated_Regularity_Levels.xlsx"  # label location

# Step 1: Load and Clean the Excel Data
print("Step 1: Loading and cleaning the Excel data...")
labels_df = pd.read_excel(file_path)
# Configuration for MAX_DATA_POINTS
MAX_DATA_POINTS = None  # Set to the desired number of data points or None to use the entire dataset
MAX_DATA_POINTS = len(labels_df) if MAX_DATA_POINTS is None else MAX_DATA_POINTS

# Limit the dataset for testing
final_labels_df = labels_df[['Object ID (Dataset Original Object ID)', 'Final Regularity Level']]
final_labels_df = final_labels_df.sample(n=min(MAX_DATA_POINTS, len(final_labels_df)), random_state=42)  # Shuffle the entire dataset
print(f"Loaded {len(final_labels_df)} data points.")

# Step 2: Extract Features from Normalized 3D OBJ Files
def extract_features_from_obj(obj_file):
    if not os.path.exists(obj_file):
        print(f'File not found: {obj_file}')
        return [None] * 10
    try:
        mesh = trimesh.load(obj_file)
        if not hasattr(mesh, 'vertices'):
            print(f'Skipping file {obj_file}: No vertices found.{" " * 20}', end='', flush=True)
            time.sleep(0.2)
            return [None] * 10

        num_vertices = len(mesh.vertices)
        num_faces = len(mesh.faces)
        surface_area = mesh.area
        volume = mesh.volume
        bounding_box_volume = mesh.bounding_box_oriented.volume
        bbox_extents = mesh.bounding_box.extents
        aspect_ratio = max(bbox_extents) / min(bbox_extents)

        # Additional features for enhanced feature engineering
        centroid = mesh.centroid
        inertia = mesh.moment_inertia
        compactness = (abs(volume) ** (1/3)) / surface_area if volume > 0 and surface_area > 0 else 0

        return [num_vertices, num_faces, surface_area, volume, bounding_box_volume, aspect_ratio,
                centroid[0], centroid[1], centroid[2], compactness]

    except Exception as e:
        print(f'Error processing file {obj_file}: {e}{" " * 20}', end='', flush=True)
        time.sleep(0.2)
        return [None] * 10

    except Exception as e:
        print(f'\rError processing file {obj_file}: {e}{" " * 20}', end='', flush=True)
        time.sleep(0.2)
        return [None] * 6

# Extract features and labels
print("Step 2: Extracting features from 3D OBJ files...")
features = []
labels = []
count = 0

for idx, row in final_labels_df.iterrows():
    obj_dir = os.path.join(base_dir, row['Object ID (Dataset Original Object ID)'])
    if os.path.isdir(obj_dir):
        # Look for the normalized obj file inside the folder
        for file_name in os.listdir(obj_dir):
            if file_name.endswith('.obj'):
                obj_file = os.path.join(obj_dir, file_name)
                count += 1
                if count % 100 == 0:
                    print(f'Processing file count: {count}')
                feature = extract_features_from_obj(obj_file)
                if None not in feature:
                    features.append(feature)
                    labels.append(row['Final Regularity Level'] - 1)  # Adjust labels to be zero-indexed
                break  # Assuming there's only one normalized OBJ file per folder

# Step 3: Prepare the Data
if len(features) == 0:
    print("No valid features were extracted. Exiting...")
    exit()

print("Step 3: Preparing the data...")
features = np.array(features)
print(f'Extracted {features.shape[1]} features for each sample.')
labels = np.array(labels)

scaler = StandardScaler()
features = scaler.fit_transform(features)

# Handle NaN values by replacing them with the mean of each column
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
features = imputer.fit_transform(features)
print("Handled NaN values by imputing with column means.")
print("Data scaling completed.")

# Balance the dataset using SMOTE
print("Balancing the dataset using SMOTE...")
smote = SMOTE(random_state=42)
features, labels = smote.fit_resample(features, labels)
print("SMOTE balancing completed.")

# Split the dataset
print("Splitting the dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
print(f"Training set size: {len(y_train)}, Testing set size: {len(y_test)}")

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Step 4: Define the Neural Network Model
print("Step 4: Defining the neural network model...")
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Add batch normalization
        self.relu = nn.ELU()  # Use ELU activation for better convergence
        self.dropout = nn.Dropout(p=0.3)  # Adjust dropout to prevent overfitting
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)  # Add batch normalization
        self.relu2 = nn.ELU()  # Use ELU activation
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)  # Add batch normalization
        self.relu3 = nn.ELU()  # Use ELU activation
        self.fc4 = nn.Linear(hidden_size, num_classes)  # Output layer

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        return out

input_size = features.shape[1]
hidden_size = 256  # Increased hidden size for more model capacity
num_classes = 4
model = NeuralNet(input_size, hidden_size, num_classes)
print("Model defined.")

# Step 5: Train the Model
num_epochs = 100  # Increase the number of epochs
learning_rate = 0.0005  # Decreased learning rate for finer updates

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # Added L2 regularization
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)  # Learning rate scheduler

print("Step 5: Training the model...")
training_loss_history = []
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    training_loss_history.append(loss.item())
    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Step 6: Evaluate the Model
print("Step 6: Evaluating the model...")
with torch.no_grad():
    y_pred = model(X_test_tensor)
    _, predicted = torch.max(y_pred, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f'Accuracy: {accuracy * 100:.2f}%')

# Step 7: Report Results
print("Step 7: Reporting results...")
print("Training Loss History:")
for i, loss in enumerate(training_loss_history):
    print(f'Epoch {i+1}: Loss = {loss:.4f}')
print(f'Final Test Accuracy: {accuracy * 100:.2f}%')