import os
import pandas as pd
import numpy as np
import trimesh
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Load the dataset
input_excel_path = 'datasets/shapenetcore/label/final_regularized_labels.xlsx'
labels_df = pd.read_excel(input_excel_path)

# Directory structure
base_dir = 'datasets/shapenetcore/ShapeNetCore.v2'

# Prepare data for training
vertices_data = []
labels = []

# Iterate through the dataset with a progress bar
for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Processing OBJ files"):
    object_id = row['Object ID (Dataset Original Object ID)']
    regularity_level = row['Final Regularity Level']

    # Find model_normalized.obj file path
    obj_folder = os.path.join(base_dir, str(object_id)[:8], object_id, 'models')
    obj_file_path = os.path.join(obj_folder, 'model_normalized.obj')

    # Check if the file exists
    if os.path.exists(obj_file_path):
        try:
            # Load the OBJ file and get vertices
            mesh = trimesh.load(obj_file_path, force='mesh')
            if isinstance(mesh, trimesh.Trimesh):
                vertices = mesh.vertices.flatten()
                vertices_data.append(vertices)
                labels.append(regularity_level)
        except Exception as e:
            print(f"Error loading file {obj_file_path}: {e}")

# Standardize the data
scaler = StandardScaler()
vertices_data = scaler.fit_transform(np.array(vertices_data))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(vertices_data, labels, test_size=0.2, random_state=42)

# Train models and evaluate accuracy
models = {
    'SVM': SVC(),
    'RandomForest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(),
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {model_name}: {accuracy:.2f}")