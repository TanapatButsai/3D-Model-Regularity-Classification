import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.preprocessing import LabelBinarizer
import trimesh
from tqdm import tqdm

# Load labels from Excel file
label_file = 'datasets/IKEA/label/Final_Validated_Regularity_Levels.xlsx'
label_data = pd.read_excel(label_file)

# Extract relevant columns
labels = label_data[['Object ID (Dataset Original Object ID)', 'Final Regularity Level', 'FolderName']]

# Path to the folder containing 3D objects
obj_folder = 'datasets/IKEA/obj-IKEA'

# Feature extraction function
def extract_features_from_obj(file_path):
    try:
        # Load the file, prioritizing loading it directly as a mesh
        loaded_obj = trimesh.load(file_path, force='mesh')  # Attempt to force load as mesh
        
        # Check if the object is a valid mesh with vertices
        if isinstance(loaded_obj, trimesh.Trimesh) and len(loaded_obj.vertices) > 0:
            # Extract features: number of vertices, number of faces, surface area, and volume
            num_vertices = len(loaded_obj.vertices)
            num_faces = len(loaded_obj.faces)
            surface_area = loaded_obj.area
            volume = loaded_obj.volume
            return [num_vertices, num_faces, surface_area, volume]
        
        print(f"Skipping file {file_path}: No valid mesh data found.")
        return None  # Skip if no valid mesh data is found

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Prepare dataset
features = []
targets = []

for index, row in tqdm(labels.iterrows(), total=len(labels)):
    object_id = row['Object ID (Dataset Original Object ID)']
    regularity_level = row['Final Regularity Level']
    folder_name = row['FolderName']
    
    # Construct the path using FolderName and Object ID
    obj_file = os.path.join(obj_folder, folder_name.strip(), object_id.strip(), 'ikea_model.obj')
    
    # Extract features
    if os.path.isfile(obj_file):
        feature_vector = extract_features_from_obj(obj_file)
        if feature_vector is not None:
            features.append(feature_vector)
            targets.append(regularity_level)

# Convert to numpy arrays
if len(features) == 0:
    print("No features extracted. Please check the dataset and feature extraction process.")
    exit()

X = np.array(features)
y = np.array(targets) - np.min(targets)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
svm_clf = SVC(probability=True)
rf_clf = RandomForestClassifier()
xgb_clf = XGBClassifier()

# Train and evaluate SVM
svm_clf.fit(X_train, y_train)
svm_predictions = svm_clf.predict(X_test)
svm_proba = svm_clf.predict_proba(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print(f"SVM Accuracy: {svm_accuracy:.2f}")

# Train and evaluate Random Forest
rf_clf.fit(X_train, y_train)
rf_predictions = rf_clf.predict(X_test)
rf_proba = rf_clf.predict_proba(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")

# Train and evaluate XGBoost
xgb_clf.fit(X_train, y_train)
xgb_predictions = xgb_clf.predict(X_test)
xgb_proba = xgb_clf.predict_proba(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_predictions)
print(f"XGBoost Accuracy: {xgb_accuracy:.2f}")

# Function to evaluate and print metrics
def evaluate_model(y_true, y_pred, y_proba=None, model_name="Model"):
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix for {model_name}:\n", conf_matrix)
    
    # Precision, Recall, F1 Score with zero_division=0 to handle undefined metrics
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    print(f"{model_name} Precision: {precision:.2f}")
    print(f"{model_name} Recall (Sensitivity): {recall:.2f}")
    print(f"{model_name} F1 Score: {f1:.2f}")
    
    # AUC-ROC and Log Loss (requires probability estimates)
    if y_proba is not None:
        lb = LabelBinarizer()
        y_true_binarized = lb.fit_transform(y_true)
        auc_roc = roc_auc_score(y_true_binarized, y_proba, average="weighted", multi_class="ovr")
        logloss = log_loss(y_true, y_proba)
        print(f"{model_name} AUC-ROC: {auc_roc:.2f}")
        print(f"{model_name} Log Loss: {logloss:.2f}")
    else:
        print(f"{model_name} AUC-ROC and Log Loss cannot be computed without probability estimates.")

# Evaluate each model
evaluate_model(y_test, svm_predictions, svm_proba, model_name="SVM")
evaluate_model(y_test, rf_predictions, rf_proba, model_name="Random Forest")
evaluate_model(y_test, xgb_predictions, xgb_proba, model_name="XGBoost")