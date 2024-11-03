import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import trimesh
from tqdm import tqdm

# Load labels from Excel file
label_file = 'datasets/ShapeNetCoreV2/label/Final_Validated_Regularity_Levels.xlsx'
label_data = pd.read_excel(label_file)

# Extract relevant columns
labels = label_data[['Object ID (Dataset Original Object ID)', 'Final Regularity Level', 'Folder Name']]

# Path to the folder containing 3D objects
obj_folder = 'datasets\ShapeNetCoreV2\obj-ShapeNetCoreV2'

# Feature extraction function
def extract_features_from_obj(file_path):
    try:
        mesh = trimesh.load(file_path)
        # Example feature extraction: number of vertices, number of faces, surface area, and volume
        num_vertices = len(mesh.vertices)
        num_faces = len(mesh.faces)
        surface_area = mesh.area
        volume = mesh.volume
        return [num_vertices, num_faces, surface_area, volume]
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Prepare dataset
features = []
targets = []

for index, row in tqdm(labels.iterrows(), total=len(labels)):
    first_layer_folder = str(int(row['Folder Name'])).zfill(8)
    second_layer_folder = str(row['Object ID (Dataset Original Object ID)']).strip()
    obj_filename = 'model_normalized'
    obj_file = os.path.join(obj_folder, first_layer_folder, second_layer_folder, 'models', f"{obj_filename}.obj")
    
    # Extract features
    if os.path.isfile(obj_file):
        feature_vector = extract_features_from_obj(obj_file)
        if feature_vector is not None:
            features.append(feature_vector)
            targets.append(row['Final Regularity Level'])

# Convert features and labels to numpy arrays
features = np.array(features)
labels = np.array(targets)

# Standardize the features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Adjust labels for XGBoost (ensure labels are from 0 to n_classes-1)
labels = labels - 1

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Define the models
models = {
    'SVM': SVC(),
    'RandomForest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(num_class=len(np.unique(y_train)))
}

# Train and evaluate each model
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{model_name} Accuracy: {accuracy:.2f}')
