import os
import pandas as pd
import numpy as np
import trimesh
from tqdm import tqdm

# Load the raw label data
file_path = 'datasets/3d-future-dataset/label/3D-FUTURE-Layout.xlsx'  # Path to the original Excel file
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

# Remove rows with NaN values in 'Final Regularity Level'
labels_df = labels_df.dropna(subset=['Final Regularity Level'])

# Convert 'Final Regularity Level' to integer and remove rows where the level is 0
labels_df['Final Regularity Level'] = labels_df['Final Regularity Level'].astype(int)
labels_df = labels_df[labels_df['Final Regularity Level'] > 0]

# Display the length of data before cleaning
print(f"Number of data points before cleaning: {len(labels_df)}")

# Function to load and validate OBJ files (only meshes, not scenes)
def load_and_validate_obj_file(file_path):
    """
    Load and validate an OBJ file to check for integrity.
    Returns vertices if the file is a valid mesh, otherwise returns None.
    """
    try:
        mesh = trimesh.load(file_path)
        # Ignore scenes, process only single mesh objects
        if isinstance(mesh, trimesh.Scene):
            print(f"Skipping file {file_path}: Scene objects are ignored.")
            return None
        vertices = mesh.vertices
        if len(vertices) == 0:
            print(f"Skipping file {file_path}: No vertices found.")
            return None
        return vertices
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def process_obj_files(base_dir, labels_df, augment=False):
    """
    Process all OBJ files with a progress bar, validate them, and apply augmentations if specified.
    """
    validated_labels = []

    for index, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Processing OBJ files", leave=True):
        obj_id = row['Object ID (Dataset Original Object ID)']
        obj_file_path = os.path.join(base_dir, obj_id, 'normalized_model.obj')

        vertices = load_and_validate_obj_file(obj_file_path)
        if vertices is not None:
            validated_labels.append(row)

    # Create a new DataFrame with validated labels only
    validated_labels_df = pd.DataFrame(validated_labels)
    return validated_labels_df

def save_validated_data(validated_labels_df, output_file_path='datasets/3d-future-dataset/label/Final_Validated_Regularity_Levels.xlsx'):
    """
    Save the validated labels to an Excel file.
    """
    validated_labels_df.to_excel(output_file_path, index=False)
    print(f"Validated data saved to {output_file_path}")

# Main script execution
if __name__ == "__main__":
    base_dir = 'datasets/3d-future-dataset/obj-3d-future-dataset'

    # Process the dataset to validate OBJ files
    validated_labels_df = process_obj_files(base_dir, labels_df, augment=True)

    # Save the validated labels to the final Excel file
    save_validated_data(validated_labels_df, 'datasets/3d-future-dataset/label/Final_Validated_Regularity_Levels.xlsx')
