import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import trimesh
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

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

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize XGBoost classifier
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Perform Stratified K-Fold Cross-Validation
fold = 1
accuracies = []

for train_index, test_index in skf.split(X, y):
    print(f"Training fold {fold}...")
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    try:
        # Train the XGBoost classifier
        xgb_clf.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = xgb_clf.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        
        print(f"Fold {fold} Accuracy: {accuracy:.2f}")
    except ValueError as e:
        print(f"Error in fold {fold}: {e}")
    
    fold += 1

# Calculate and print the average accuracy across all folds
if len(accuracies) > 0:
    average_accuracy = np.mean(accuracies)
    print(f"\nAverage Cross-Validation Accuracy: {average_accuracy:.2f}")
else:
    print("No valid folds were completed.")