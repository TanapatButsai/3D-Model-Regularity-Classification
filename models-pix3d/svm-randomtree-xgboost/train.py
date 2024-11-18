import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score,
                             f1_score, roc_auc_score, log_loss)
import trimesh
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

# Load labels from Excel file
label_file = 'datasets/pix3d/label/Final_Validated_Regularity_Levels.xlsx'
label_data = pd.read_excel(label_file)

# Extract relevant columns
labels = label_data[['Object ID (Dataset Original Object ID)', 'Final Regularity Level']]

# Path to the folder containing 3D objects
obj_folder = 'datasets/pix3d/obj-pix3d'

# Feature extraction function that handles Scene objects
def extract_features_from_obj(file_path):
    try:
        mesh = trimesh.load(file_path)
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) > 0:
                mesh = list(mesh.geometry.values())[0]
            else:
                print(f"Skipping file {file_path}: No valid geometry found.")
                return None
        if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces'):
            print(f"Skipping file {file_path}: Not a valid mesh object.")
            return None
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
    object_id = row['Object ID (Dataset Original Object ID)']
    regularity_level = row['Final Regularity Level']
    obj_file = os.path.join(obj_folder, 'models', object_id.strip(), 'model.obj')
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
y = np.array(targets)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
num_classes = len(np.unique(y))

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
models = {
    "SVM": SVC(probability=True),
    "RandomForest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

# Directory to save results
results_dir = "datasets/pix3d"
os.makedirs(results_dir, exist_ok=True)

# Function to plot and save confusion matrix
def save_confusion_matrix(conf_matrix, model_name, labels, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# One-hot encode y_test for AUC-ROC calculation
encoder = OneHotEncoder(sparse=False)
y_test_onehot = encoder.fit_transform(y_test.reshape(-1, 1))

# Train and evaluate each model
all_metrics = []
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Ensure probabilities have correct shape
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)
        if y_pred_proba.shape[1] != num_classes:
            full_probs = np.zeros((y_pred_proba.shape[0], num_classes))
            full_probs[:, :y_pred_proba.shape[1]] = y_pred_proba
            y_pred_proba = full_probs
    else:
        y_pred_proba = None

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
    
    if y_pred_proba is not None:
        try:
            # Use one-hot encoded y_test for AUC-ROC
            auc_roc = roc_auc_score(y_test_onehot, y_pred_proba, multi_class='ovr', average="weighted")
        except ValueError as e:
            print(f"AUC-ROC calculation failed for {model_name}: {e}")
            auc_roc = "N/A"
        try:
            logloss = log_loss(y_test, y_pred_proba, labels=np.arange(num_classes))
        except ValueError:
            logloss = "N/A"
    else:
        auc_roc, logloss = "N/A", "N/A"
    
    # Save metrics
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "AUC-ROC": auc_roc,
        "Log Loss": logloss
    }
    all_metrics.append(metrics)
    
    # Save confusion matrix as image
    conf_matrix_image_path = os.path.join(results_dir, f"{model_name}_confusion_matrix.png")
    save_confusion_matrix(conf_matrix, model_name, label_encoder.classes_, conf_matrix_image_path)
    print(f"Confusion matrix image saved for {model_name} at {conf_matrix_image_path}")