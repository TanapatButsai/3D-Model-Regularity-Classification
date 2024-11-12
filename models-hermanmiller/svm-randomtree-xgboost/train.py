import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, 
    recall_score, f1_score, roc_auc_score, log_loss
)
from sklearn.preprocessing import LabelBinarizer
import trimesh
from tqdm import tqdm

# Load labels from Excel file
label_file = 'datasets/hermanmiller/label/Final_Validated_Regularity_Levels.xlsx'
label_data = pd.read_excel(label_file)

# Extract relevant columns
labels = label_data[['Object ID (Dataset Original Object ID)', 'Final Regularity Level']]

# Path to the folder containing 3D objects
obj_folder = 'datasets/hermanmiller/obj-hermanmiller'

# Feature extraction function
def extract_features_from_obj(file_path):
    try:
        mesh = trimesh.load(file_path, force='mesh')
        
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) > 0:
                mesh = next(iter(mesh.geometry.values()))
            else:
                print(f"Skipping file {file_path}: No mesh data in scene.")
                return None
        
        if isinstance(mesh, trimesh.Trimesh):
            num_vertices = len(mesh.vertices)
            num_faces = len(mesh.faces)
            surface_area = mesh.area
            volume = mesh.volume
            return [num_vertices, num_faces, surface_area, volume]
        else:
            print(f"Skipping file {file_path}: Not a valid mesh object.")
            return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Prepare dataset
features = []
targets = []

for index, row in tqdm(labels.iterrows(), total=len(labels)):
    object_id = row['Object ID (Dataset Original Object ID)']
    regularity_level = row['Final Regularity Level']
    
    obj_file = os.path.join(obj_folder, object_id.strip(), f"{object_id.strip()}.obj")
    
    if os.path.isfile(obj_file):
        feature_vector = extract_features_from_obj(obj_file)
        if feature_vector is not None:
            features.append(feature_vector)
            targets.append(regularity_level)

if len(features) == 0:
    print("No features extracted. Please check the dataset and feature extraction process.")
    exit()

X = np.array(features)
y = np.array(targets) - 1

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
svm_clf = SVC(probability=True)
rf_clf = RandomForestClassifier()
xgb_clf = XGBClassifier()

# Function to evaluate and print metrics
def evaluate_model(y_true, y_pred, y_proba=None, model_name="Model"):
    print(f"\nMetrics for {model_name}:")
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", conf_matrix)
    
    # Precision, Recall, F1 Score with zero_division=1 to avoid warnings
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=1)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=1)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=1)
    
    print(f"Precision: {precision:.2f}")
    print(f"Recall (Sensitivity): {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    
    # AUC-ROC and Log Loss (requires probability estimates)
    if y_proba is not None:
        lb = LabelBinarizer()
        y_true_binarized = lb.fit_transform(y_true)
        auc_roc = roc_auc_score(y_true_binarized, y_proba, average="weighted", multi_class="ovr")
        logloss = log_loss(y_true, y_proba)
        print(f"AUC-ROC: {auc_roc:.2f}")
        print(f"Log Loss: {logloss:.2f}")
    else:
        print("AUC-ROC and Log Loss cannot be computed without probability estimates.")

# Train and evaluate SVM
svm_clf.fit(X_train, y_train)
svm_predictions = svm_clf.predict(X_test)
svm_proba = svm_clf.predict_proba(X_test)
evaluate_model(y_test, svm_predictions, svm_proba, model_name="SVM")

# Train and evaluate Random Forest
rf_clf.fit(X_train, y_train)
rf_predictions = rf_clf.predict(X_test)
rf_proba = rf_clf.predict_proba(X_test)
evaluate_model(y_test, rf_predictions, rf_proba, model_name="Random Forest")

# Train and evaluate XGBoost
xgb_clf.fit(X_train, y_train)
xgb_predictions = xgb_clf.predict(X_test)
xgb_proba = xgb_clf.predict_proba(X_test)
evaluate_model(y_test, xgb_predictions, xgb_proba, model_name="XGBoost")
