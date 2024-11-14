import os
import pandas as pd
import numpy as np
import trimesh
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import torch
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*use_label_encoder*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*tree method `gpu_hist` is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*The y_pred values do not sum to one.*")

# Configuration
base_dir = 'datasets/3d-future-dataset/obj-3d.future'
file_path = "datasets/3d-future-dataset/label/Final_Validated_Regularity_Levels.xlsx"

# Check for CUDA availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on: {device.upper()}")

# Load and preprocess the data
labels_df = pd.read_excel(file_path)
MAX_DATA_POINTS = 10000
final_labels_df = labels_df[['Object ID (Dataset Original Object ID)', 'Final Regularity Level']]
final_labels_df = final_labels_df.sample(n=min(MAX_DATA_POINTS, len(final_labels_df)), random_state=42)

# Adjust labels to start from 0
final_labels_df['Final Regularity Level'] -= 1

# Feature extraction function
def extract_features_from_obj(obj_file):
    try:
        mesh = trimesh.load(obj_file, force='mesh')
        if not isinstance(mesh, trimesh.Trimesh):
            return None  # Skip non-mesh files
        num_vertices = len(mesh.vertices)
        num_faces = len(mesh.faces)
        surface_area = mesh.area
        volume = mesh.volume
        bounding_box_volume = mesh.bounding_box_oriented.volume
        bbox_extents = mesh.bounding_box.extents
        aspect_ratio = max(bbox_extents) / min(bbox_extents)
        return [num_vertices, num_faces, surface_area, volume, bounding_box_volume, aspect_ratio]
    except Exception:
        return None

# Extract features
features_list = []
object_ids = []

print(f"Processing {len(final_labels_df)} OBJ Files...")
for index, row in tqdm(final_labels_df.iterrows(), total=len(final_labels_df), desc="Processing OBJ Files"):
    obj_id = row['Object ID (Dataset Original Object ID)']
    model_folder = os.path.join(base_dir, str(obj_id))
    normalized_model_file = os.path.join(model_folder, 'normalized_model.obj')

    if os.path.exists(normalized_model_file):
        features = extract_features_from_obj(normalized_model_file)
        if features:
            features_list.append(features)
            object_ids.append(obj_id)

features_df = pd.DataFrame(features_list, columns=['Num Vertices', 'Num Faces', 'Surface Area', 'Volume', 'Bounding Box Volume', 'Aspect Ratio'])
final_labels_df = final_labels_df[final_labels_df['Object ID (Dataset Original Object ID)'].isin(object_ids)]
dataset_df = pd.concat([features_df, final_labels_df['Final Regularity Level'].reset_index(drop=True)], axis=1)
dataset_df = dataset_df.dropna()

# Preprocess and balance data
X = dataset_df.drop('Final Regularity Level', axis=1)
y = dataset_df['Final Regularity Level']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Adjust k_neighbors based on the smallest class size
from collections import Counter
class_counts = Counter(y)
min_class_size = min(class_counts.values())
k_neighbors = min(5, max(1, min_class_size - 1))

# Apply SMOTE
smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Move data to GPU if available
X_train_np = torch.tensor(X_train, dtype=torch.float32).to(device).cpu().numpy()
X_test_np = torch.tensor(X_test, dtype=torch.float32).to(device).cpu().numpy()
y_train_np = torch.tensor(y_train.values, dtype=torch.long).to(device).cpu().numpy()
y_test_np = torch.tensor(y_test.values, dtype=torch.long).to(device).cpu().numpy()

# Hyperparameter Tuning for XGBoost and RandomForest
tuned_params_rf = {'n_estimators': [100, 300], 'max_depth': [10, 30, None]}
tuned_params_xgb = {'n_estimators': [100, 300], 'learning_rate': [0.01, 0.1], 'max_depth': [10, 20]}

print("\nTuning RandomForest...")
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), tuned_params_rf, cv=3, scoring='accuracy')
grid_rf.fit(X_train_np, y_train_np)
best_rf = grid_rf.best_estimator_

print("\nTuning XGBoost...")
grid_xgb = GridSearchCV(XGBClassifier(eval_metric='mlogloss', device='cuda'), tuned_params_xgb, cv=3, scoring='accuracy')
grid_xgb.fit(X_train_np, y_train_np)
best_xgb = grid_xgb.best_estimator_

# Voting Ensemble
ensemble_model = VotingClassifier(estimators=[('rf', best_rf), ('xgb', best_xgb)], voting='soft')

# Models dictionary
models = {
    "SVM": SVC(probability=True, random_state=42),
    "RandomForest": best_rf,
    "XGBoost": best_xgb,
    "Ensemble": ensemble_model
}

# Fit and evaluate models
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")

    if model_name == "XGBoost":
        model.set_params(device="cuda" if device == 'cuda' else "cpu")
        model.fit(X_train_np, y_train_np)
    else:
        model.fit(X_train_np, y_train_np)

    y_pred = model.predict(X_test_np)
    y_pred_proba = model.predict_proba(X_test_np) if hasattr(model, "predict_proba") else None

    # Calculate metrics
    accuracy = accuracy_score(y_test_np, y_pred)
    precision = precision_score(y_test_np, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test_np, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_test_np, y_pred, average='weighted', zero_division=1)
    auc_roc = roc_auc_score(y_test_np, y_pred_proba, multi_class='ovr') if y_pred_proba is not None else 'N/A'
    logloss = log_loss(y_test_np, y_pred_proba) if y_pred_proba is not None else 'N/A'

    # Print results
    print(f"{model_name} Results:")
    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_np, y_pred))
    print(f"Precision: {precision:.2f}")
    print(f"Recall (Sensitivity): {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"AUC-ROC: {auc_roc}")
    print(f"Log Loss: {logloss}")
