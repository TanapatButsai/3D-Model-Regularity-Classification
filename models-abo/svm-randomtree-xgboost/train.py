import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score,
                             log_loss, roc_curve, auc)
import trimesh
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load labels from Excel file
label_file = 'datasets/abo/label/Final_Validated_Regularity_Levels.xlsx'
label_data = pd.read_excel(label_file)

# Define MAX_DATA_POINTS
MAX = len(label_data)
MAX_DATA_POINTS = MAX  # You can change this number based on how many data points you want to train with

# Limit the number of data points based on MAX_DATA_POINTS
if len(label_data) > MAX_DATA_POINTS:
    label_data = label_data.sample(n=MAX_DATA_POINTS, random_state=42)

# Extract relevant columns
labels = label_data[['Object ID (Dataset Original Object ID)', 'Final Regularity Level']]

# Path to the folder containing 3D objects
obj_folder = 'datasets/abo/obj-ABO'

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
    obj_file = os.path.join(obj_folder, object_id.strip(), f'{object_id.strip()}.obj')

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

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
svm_clf = SVC(probability=True)
rf_clf = RandomForestClassifier()
xgb_clf = XGBClassifier(eval_metric='mlogloss')

# Function to evaluate model
def evaluate_model(clf, X_test, y_test, model_name):
    predictions = clf.predict(X_test)
    prob_predictions = clf.predict_proba(X_test) if hasattr(clf, 'predict_proba') else None

    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    auc_roc = roc_auc_score(y_test, prob_predictions, multi_class='ovr') if prob_predictions is not None else 'N/A'
    logloss = log_loss(y_test, prob_predictions) if prob_predictions is not None else 'N/A'

    print(f"\n{model_name} Evaluation:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"AUC-ROC: {auc_roc}")
    print(f"Log Loss: {logloss}")

# Train and evaluate SVM
svm_clf.fit(X_train, y_train)
evaluate_model(svm_clf, X_test, y_test, "SVM")

# Train and evaluate Random Forest
rf_clf.fit(X_train, y_train)
evaluate_model(rf_clf, X_test, y_test, "Random Forest")

# Train and evaluate XGBoost
try:
    xgb_clf.fit(X_train, y_train)
    evaluate_model(xgb_clf, X_test, y_test, "XGBoost")
except ValueError as e:
    print(f"XGBoost training error: {e}")
