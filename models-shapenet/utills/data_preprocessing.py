import os
import trimesh
import pandas as pd
import numpy as np

# Path configuration
base_dir = 'datasets/shapenetcore/ShapeNetCore.v2'  # Replace with your dataset path
input_excel_path = 'datasets/shapenetcore/label/shapenet.xlsx'
output_file_path = 'datasets/shapenetcore/label/final_shapenet.xlsx'  # Path to the input Excel file

# Constants
MAX_VERTICES = 50000

# Function to load and validate OBJ file
def load_and_validate_obj_file(file_path):
    try:
        mesh = trimesh.load(file_path)
        vertices = mesh.vertices
        if len(vertices) == 0:
            print(f"Skipping file {file_path}: No vertices found.")
            return None
        # Pad or truncate vertices to ensure consistent size
        if len(vertices) > MAX_VERTICES:
            vertices = vertices[:MAX_VERTICES]
        else:
            padding = MAX_VERTICES - len(vertices)
            vertices = np.pad(vertices, ((0, padding), (0, 0)), mode='constant', constant_values=0)
        return vertices
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

# Function to determine the final regularity level
def determine_final_level(row):
    confidences = [
        ('Layout level (Person 1)', 'Layout level confident (Person 1)'),
        ('Layout level (Person 2)', 'Layout level confident (Person 2)'),
        ('Layout level (Person 3)', 'Layout level confident (Person 3)'),
        ('Layout level (Person 4)', 'Layout level confident (Person 4)'),
        ('Layout level (Person 5)', 'Layout level confident (Person 5)'),
        ('Layout level (Person 6)', 'Layout level confident (Person 6)'),
        ('Layout level (Person 7)', 'Layout level confident (Person 7)')
    ]
    valid_levels = []
    for level, confidence in confidences:
        if not pd.isna(row[confidence]) and row[confidence] == 1:
            valid_levels.append(row[level])
    
    if valid_levels:
        return round(np.mean(valid_levels))
    else:
        for level, confidence in confidences:
            if not pd.isna(row[level]):
                return row[level]
    return np.nan

# Function to process all OBJ files in the dataset
def process_dataset(base_dir, labels_df):
    data_records = []
    
    # Traverse the dataset directory
    for root, _, files in os.walk(base_dir):
        if 'model_normalized.obj' in files:
            obj_file_path = os.path.join(root, 'model_normalized.obj')
            vertices = load_and_validate_obj_file(obj_file_path)
            if vertices is not None:
                # Extract the identifier from the directory name (e.g., the parent folder name)
                identifier = os.path.basename(root)
                # Find the corresponding row in the labels DataFrame
                label_row = labels_df[labels_df['Object ID'] == identifier]
                if not label_row.empty:
                    final_level = determine_final_level(label_row.iloc[0])
                    if not pd.isna(final_level):
                        data_records.append({'Identifier': identifier, 'Vertices': vertices, 'Final Regularity Level': final_level})
    
    # Convert the data to a DataFrame and save
    if data_records:
        df = pd.DataFrame(data_records)
        df.to_excel(output_file_path, index=False)
        print(f"Processed data saved to {output_file_path}")
        print(f"Input labels count: {len(labels_df)}, Final labels count: {len(df)}")
    else:
        print("No valid OBJ files found to process.")

# Main script execution
if __name__ == "__main__":
    labels_df = pd.read_excel(input_excel_path)
    process_dataset(base_dir, labels_df)