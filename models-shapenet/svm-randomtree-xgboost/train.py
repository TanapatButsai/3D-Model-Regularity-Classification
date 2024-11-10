import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score,
                             f1_score, roc_auc_score, log_loss)
from sklearn.preprocessing import StandardScaler
import trimesh
from tqdm import tqdm

# Configuration: set the number of samples to use
num_samples = 8000  # Change this value to limit the number of samples processed

# Load labels from Excel file
label_file = 'datasets/ShapeNetCoreV2/label/final_regularized_labels.xlsx'
label_data = pd.read_excel(label_file)

# Extract relevant columns
labels = label_data[['Object ID (Dataset Original Object ID)', 'Final Regularity Level', 'Folder Name']]

# Path to the folder containing 3D objects
obj_folder = 'datasets/ShapeNetCoreV2/obj-ShapeNetCoreV2'

# Feature extraction function
def extract_features_from_obj(file_path):
    try:
        loaded_obj = trimesh.load(file_path)
        if isinstance(loaded_obj, trimesh.Scene):
            if loaded_obj.geometry:
                mesh = trimesh.util.concatenate(loaded_obj.geometry.values())
            else:
                print(f"Scene has no geometries: {file_path}")
                return None
        else:
            mesh = loaded_obj

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
skipped_files = 0  # Counter for skipped files due to errors

# Iterate over the labels and process only up to num_samples
for index, row in tqdm(labels.iterrows(), total=min(len(labels), num_samples)):
    if len(features) >= num_samples:
        break
    
    try:
        first_layer_folder = str(int(row['Folder Name'])).zfill(8)
        second_layer_folder = str(row['Object ID (Dataset Original Object ID)']).strip()
        obj_filename = 'model_normalized'
        obj_file = os.path.join(obj_folder, first_layer_folder, second_layer_folder, 'models', f"{obj_filename}.obj")
        
        if os.path.isfile(obj_file):
            feature_vector = extract_features_from_obj(obj_file)
            if feature_vector is not None:
                features.append(feature_vector)
                targets.append(row['Final Regularity Level'])
            else:
                skipped_files += 1
        else:
            print(f"File not found: {obj_file}")
            skipped_files += 1
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        skipped_files += 1

features = np.array(features)
targets = np.array(targets)

scaler = StandardScaler()
features = scaler.fit_transform(features)

targets = targets - 1
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

models = {
    'SVM': SVC(probability=True),  # Enable probability estimates for AUC-ROC and Log Loss
    'RandomForest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(num_class=len(np.unique(y_train)))
}

# Train and evaluate each model
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{model_name} Accuracy: {accuracy:.2f}')

    # Calculate additional metrics
    conf_matrix = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    auc_roc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr') if y_pred_proba is not None else 'N/A'
    logloss = log_loss(y_test, y_pred_proba) if y_pred_proba is not None else 'N/A'

    # Print metrics
    print(f'{model_name} Confusion Matrix:\n{conf_matrix}')
    print(f'{model_name} Precision: {precision:.2f}')
    print(f'{model_name} Recall (Sensitivity): {recall:.2f}')
    print(f'{model_name} F1 Score: {f1:.2f}')
    print(f'{model_name} AUC-ROC: {auc_roc}')
    print(f'{model_name} Log Loss: {logloss}')

print(f"Total files skipped due to errors: {skipped_files}")
