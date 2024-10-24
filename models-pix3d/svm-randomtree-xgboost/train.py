import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import trimesh
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# Load labels from Excel file
label_file = 'datasets/pix3d/label/Final_Validated_Regularity_Levels.xlsx'
label_data = pd.read_excel(label_file)

# Extract relevant columns
labels = label_data[['Object ID (Dataset Original Object ID)', 'Final Regularity Level']]

# Path to the folder containing 3D objects
obj_folder = 'datasets/pix3d/obj-pix3d'

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
    object_id = row['Object ID (Dataset Original Object ID)']
    regularity_level = row['Final Regularity Level']
    
    # Construct the path
    obj_file = os.path.join(obj_folder, 'models', object_id.strip(), 'model.obj')
    
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
y = np.array(targets)

# Label encoding to ensure classes are continuous and start from 0
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Check for unique labels and print them
unique_labels = np.unique(y)
print("Unique labels after encoding:", unique_labels)

# Ensure there are no missing labels (continuous from 0)
expected_labels = np.arange(len(unique_labels))
if not np.array_equal(unique_labels, expected_labels):
    print("Warning: Missing labels detected. Expected labels:", expected_labels, "but got:", unique_labels)
    # Optional: Consider filling missing labels with dummy samples or remove labels without enough samples

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
svm_clf = SVC()
rf_clf = RandomForestClassifier()
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Train and evaluate SVM
svm_clf.fit(X_train, y_train)
svm_predictions = svm_clf.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print(f"SVM Accuracy: {svm_accuracy:.2f}")

# Train and evaluate Random Forest
rf_clf.fit(X_train, y_train)
rf_predictions = rf_clf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")

# Train and evaluate XGBoost
try:
    xgb_clf.fit(X_train, y_train)
    xgb_predictions = xgb_clf.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_predictions)
    print(f"XGBoost Accuracy: {xgb_accuracy:.2f}")
except ValueError as e:
    print(f"XGBoost training error: {e}")

# Compare model accuracies
print("\nModel Comparison:")
print(f"SVM Accuracy: {svm_accuracy:.2f}")
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
if 'xgb_accuracy' in locals():
    print(f"XGBoost Accuracy: {xgb_accuracy:.2f}")