import os
import pandas as pd
import numpy as np
import trimesh
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score, log_loss
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
from sklearn.ensemble import VotingClassifier

# Configuration
base_dir = 'datasets/3d-future-dataset/obj-3d-future-dataset'
file_path = "datasets/3d-future-dataset/label/Final_Validated_Regularity_Levels.xlsx"

# Load and preprocess the data
labels_df = pd.read_excel(file_path)
MAX_DATA_POINTS = 3000
final_labels_df = labels_df[['Object ID (Dataset Original Object ID)', 'Final Regularity Level']]
final_labels_df = final_labels_df.sample(n=min(MAX_DATA_POINTS, len(final_labels_df)), random_state=42)

# Feature extraction function
def extract_features_from_obj(obj_file):
    try:
        mesh = trimesh.load(obj_file)
        if not hasattr(mesh, 'vertices'):
            return [None] * 6

        num_vertices = len(mesh.vertices)
        num_faces = len(mesh.faces)
        surface_area = mesh.area
        volume = mesh.volume
        bounding_box_volume = mesh.bounding_box_oriented.volume
        bbox_extents = mesh.bounding_box.extents
        aspect_ratio = max(bbox_extents) / min(bbox_extents)
        
        return [num_vertices, num_faces, surface_area, volume, bounding_box_volume, aspect_ratio]

    except Exception:
        return [None] * 6

# Extract features
features_list = []
object_ids = []

for index, row in tqdm(final_labels_df.iterrows(), total=len(final_labels_df), desc="Processing OBJ Files"):
    obj_id = row['Object ID (Dataset Original Object ID)']
    model_folder = os.path.join(base_dir, str(obj_id))
    normalized_model_file = os.path.join(model_folder, 'normalized_model.obj')

    if os.path.exists(normalized_model_file):
        features = extract_features_from_obj(normalized_model_file)
        features_list.append(features)
        object_ids.append(obj_id)

features_df = pd.DataFrame(features_list, columns=['Num Vertices', 'Num Faces', 'Surface Area', 'Volume', 'Bounding Box Volume', 'Aspect Ratio'])
final_labels_df = final_labels_df[final_labels_df['Object ID (Dataset Original Object ID)'].isin(object_ids)]
dataset_df = pd.concat([features_df, final_labels_df['Final Regularity Level']], axis=1)
dataset_df = dataset_df.dropna()

# Preprocess and balance data
X = dataset_df.drop('Final Regularity Level', axis=1)
y = dataset_df['Final Regularity Level']
scaler = StandardScaler()
X = scaler.fit_transform(X)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Hyperparameter Tuning for XGBoost and RandomForest
tuned_params_rf = {
    'n_estimators': [100, 300],
    'max_depth': [10, 30, None]
}
tuned_params_xgb = {
    'n_estimators': [100, 300],
    'learning_rate': [0.01, 0.1],
    'max_depth': [10, 20]
}

print("\nTuning RandomForest...")
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), tuned_params_rf, cv=3, scoring='accuracy')
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_

print("\nTuning XGBoost...")
grid_xgb = GridSearchCV(XGBClassifier(eval_metric='mlogloss', use_label_encoder=False), tuned_params_xgb, cv=3, scoring='accuracy')
grid_xgb.fit(X_train, y_train)
best_xgb = grid_xgb.best_estimator_

# Voting Ensemble
ensemble_model = VotingClassifier(estimators=[('rf', best_rf), ('xgb', best_xgb)], voting='soft')

# Fit and evaluate models
for model_name, model in [("SVM", SVC(probability=True)), ("RandomForest", best_rf), ("XGBoost", best_xgb), ("Ensemble", ensemble_model)]:
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
    auc_roc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr') if y_pred_proba is not None else 'N/A'
    logloss = log_loss(y_test, y_pred_proba) if y_pred_proba is not None else 'N/A'

    # Print results
    print(f"{model_name} Results:")
    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"Precision: {precision:.2f}")
    print(f"Recall (Sensitivity): {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"AUC-ROC: {auc_roc}")
    print(f"Log Loss: {logloss}")
