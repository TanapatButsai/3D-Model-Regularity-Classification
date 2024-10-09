import pandas as pd
import trimesh
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

MAX_DATA_POINTS = 5000
base_dir = 'datasets/3d-future-dataset/3D-FUTURE-model'  # Replace with the path to the folder that contains all model subfolders
file_path = "datasets/3d-future-dataset/label/Final_Validated_Regularity_Levels.xlsx"
# Step 1: Load and Clean the Excel Data

labels_df = pd.read_excel(file_path)

# Function to determine the final regularity level
def determine_final_level(row):
    if pd.isna(row['Layout level (Person 1)']):
        return row['Layout level (Person 2)']
    elif pd.isna(row['Layout level (Person 2)']):
        return row['Layout level (Person 1)']
    
    if row['Layout level confident (Person 1)'] > row['Layout level confident (Person 2)']:
        return row['Layout level (Person 1)']
    elif row['Layout level confident (Person 2)'] > row['Layout level confident (Person 1)']:
        return row['Layout level (Person 2)']
    
    if row['Layout level confident (Person 1)'] == 1 and row['Layout level confident (Person 2)'] == 1:
        return round((row['Layout level (Person 1)'] + row['Layout level (Person 2)']) / 2)

    return row['Layout level (Person 1)']

# Apply the function to create the 'Final Regularity Level' column
labels_df['Final Regularity Level'] = labels_df.apply(determine_final_level, axis=1)

# Save the final labels to a new dataframe
final_labels_df = labels_df[['Object ID (Dataset Original Object ID)', 'Final Regularity Level']]

# Limit the dataset to the first 500 rows for testing
# final_labels_df = final_labels_df.head(MAX_DATA_POINTS))
final_labels_df = final_labels_df.sample(n=min(MAX_DATA_POINTS, len(final_labels_df)), random_state=42)
# # Specify the output file path
# output_file_path = 'data/label/Final_Regularity_Levels.xlsx'

# # Save the DataFrame to a new Excel file
# final_labels_df.to_excel(output_file_path, index=False)

# Step 2: Extract Features from the Normalized 3D OBJ Files
def extract_features_from_obj(obj_file):
    try:
        # Load the file with trimesh
        mesh = trimesh.load(obj_file)

        # Check if the mesh object has vertices
        if not hasattr(mesh, 'vertices'):
            print(f"Skipping file {obj_file}: No vertices found.")
            return [None, None, None, None, None, None]

        # Extract features if the vertices exist
        num_vertices = len(mesh.vertices)
        num_faces = len(mesh.faces)
        surface_area = mesh.area
        volume = mesh.volume
        bounding_box_volume = mesh.bounding_box_oriented.volume
        bbox_extents = mesh.bounding_box.extents
        aspect_ratio = max(bbox_extents) / min(bbox_extents)
        
        # # Debug: Print extracted features for the current file
        # print(f"File: {obj_file}")
        # print(f"Vertices: {num_vertices}, Faces: {num_faces}, Surface Area: {surface_area}, Volume: {volume}, Bounding Box Volume: {bounding_box_volume}, Aspect Ratio: {aspect_ratio}")
        
        return [num_vertices, num_faces, surface_area, volume, bounding_box_volume, aspect_ratio]

    except Exception as e:
        print(f"Error processing file {obj_file}: {e}")
        return [None, None, None, None, None, None]

# Directory where the folders containing the OBJ files are stored
features_list = []
object_ids = []

# Loop through each folder and extract features from the normalized_model.obj file
for index, row in final_labels_df.iterrows():
    model_folder = os.path.join(base_dir, row['Object ID (Dataset Original Object ID)'])
    normalized_model_file = os.path.join(model_folder, 'normalized_model.obj')
    
    if os.path.exists(normalized_model_file):
        features = extract_features_from_obj(normalized_model_file)
        features_list.append(features)
        object_ids.append(row['Object ID (Dataset Original Object ID)'])

# Convert the features list into a DataFrame
features_df = pd.DataFrame(features_list, columns=['Num Vertices', 'Num Faces', 'Surface Area', 'Volume', 'Bounding Box Volume', 'Aspect Ratio'])

# Step 3: Prepare the Dataset
# Combine features with final labels
dataset_df = pd.concat([features_df, final_labels_df['Final Regularity Level']], axis=1)

# Remove rows with None values in features before model training
dataset_df = dataset_df.dropna()

# Step 4: Train a Machine Learning Model
X = dataset_df.drop('Final Regularity Level', axis=1)  # Features
y = dataset_df['Final Regularity Level']  # Labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest classifier with class weights to handle imbalance
print("Start")
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Step 5: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print classification report with zero division handling
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))