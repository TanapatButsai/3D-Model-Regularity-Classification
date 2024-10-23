import pandas as pd
import trimesh
import os
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Configuration
base_dir = 'datasets/3d-future-dataset/3D-FUTURE-model' # obj dir location
file_path = "datasets/3d-future-dataset/label/Final_Validated_Regularity_Levels.xlsx" #label location

# Step 1: Load and Clean the Excel Data
labels_df = pd.read_excel(file_path)
MAX_DATA_POINTS = 3000

# Limit the dataset for testing
final_labels_df = labels_df[['Object ID (Dataset Original Object ID)', 'Final Regularity Level']]
final_labels_df = final_labels_df.sample(n=min(MAX_DATA_POINTS, len(final_labels_df)), random_state=42)

# Step 2: Extract Features from Normalized 3D OBJ Files
def extract_features_from_obj(obj_file):
    try:
        mesh = trimesh.load(obj_file)
        if not hasattr(mesh, 'vertices'):
            print(f'\rSkipping file {obj_file}: No vertices found.{" " * 20}', end='', flush=True)
            time.sleep(0.2)
            return [None] * 6

        num_vertices = len(mesh.vertices)
        num_faces = len(mesh.faces)
        surface_area = mesh.area
        volume = mesh.volume
        bounding_box_volume = mesh.bounding_box_oriented.volume
        bbox_extents = mesh.bounding_box.extents
        aspect_ratio = max(bbox_extents) / min(bbox_extents)
        
        return [num_vertices, num_faces, surface_area, volume, bounding_box_volume, aspect_ratio]

    except Exception as e:
        print(f'\rError processing file {obj_file}: {e}{" " * 20}', end='', flush=True)
        time.sleep(0.2)
        return [None] * 6

# Custom loading bar function with updates every 0.1%, starting at 0%
def loading_bar(iteration, total, prefix='', suffix='', length=50, fill='#'):
    percent = 100 * (iteration / float(total))
    if percent % 0.1 < (100 / total) or iteration == total or iteration == 1:
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        print(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}', end='', flush=True)
        if iteration == total:
            print()


features_list = []
object_ids = []

# Loop through each folder and extract features from the normalized_model.obj file
total_rows = len(final_labels_df)
print(f"Training Data: {MAX_DATA_POINTS}")
for index, row in enumerate(final_labels_df.iterrows(), start=1):
    loading_bar(index, total_rows, prefix="Extracting Features from OBJ Files")

    _, row_data = row
    model_folder = os.path.join(base_dir, str(row_data['Object ID (Dataset Original Object ID)']))
    normalized_model_file = os.path.join(model_folder, 'normalized_model.obj')

    if os.path.exists(normalized_model_file):
        features = extract_features_from_obj(normalized_model_file)
        features_list.append(features)
        object_ids.append(row_data['Object ID (Dataset Original Object ID)'])

# Print a final new line to ensure the console looks clean after progress
print()

features_df = pd.DataFrame(features_list, columns=['Num Vertices', 'Num Faces', 'Surface Area', 'Volume', 'Bounding Box Volume', 'Aspect Ratio'])
final_labels_df = final_labels_df[final_labels_df['Object ID (Dataset Original Object ID)'].isin(object_ids)]
dataset_df = pd.concat([features_df, final_labels_df['Final Regularity Level']], axis=1)
dataset_df = dataset_df.dropna()

# Step 3: Preprocessing and Balancing
X = dataset_df.drop('Final Regularity Level', axis=1)
y = dataset_df['Final Regularity Level']

scaler = StandardScaler()
X = scaler.fit_transform(X)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Step 4: Hyperparameter Tuning with GridSearchCV
clf = RandomForestClassifier(random_state=42, class_weight='balanced')

param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Setting verbose to 0 to suppress GridSearchCV output
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=2, n_jobs=-1, verbose=0)

# Manually simulate the progress of GridSearchCV with a custom loading bar
total_fits = len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * len(param_grid['max_features']) * 2  # `cv=2`
fits_done = 0

# Start the loading bar for training and tuning progress
loading_bar(fits_done, total_fits, prefix="Training and Tuning Model")

# Loop through the parameter grid and manually fit while updating the loading bar
for params in grid_search.param_grid['n_estimators']:
    for max_depth in grid_search.param_grid['max_depth']:
        for min_samples_split in grid_search.param_grid['min_samples_split']:
            for min_samples_leaf in grid_search.param_grid['min_samples_leaf']:
                for max_features in grid_search.param_grid['max_features']:
                    clf.set_params(
                        n_estimators=params,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features
                    )
                    clf.fit(X_train, y_train)
                    fits_done += 1
                    loading_bar(fits_done, total_fits, prefix="Training and Tuning Model")

# Use the best estimator from the grid search (in this case, we'll use the last fit model)
best_model = clf

# Step 5: Evaluate the Model
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'\nOptimized Accuracy: {accuracy * 100:.2f}%')

print("\nOptimized Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))