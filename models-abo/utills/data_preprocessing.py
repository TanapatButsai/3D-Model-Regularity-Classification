import os
import pandas as pd
import numpy as np
import trimesh

# Load the raw label data
excel_path = 'datasets/abo/label/abo.xlsx'
data = pd.read_excel(excel_path)

data.columns = data.columns.str.replace(r'\s+', ' ', regex=True).str.strip()

def determine_final_level(row):
    # List of persons and their corresponding layout levels and confidence columns
    persons = ['Person 1', 'Person 2', 'Person 3']
    levels = []
    confidences = []

    # Extract levels and confidences if columns exist
    for person in persons:
        level_col = f'Layout level ({person})'
        conf_col = f'Layout level confident ({person})'

        if level_col in row.index and conf_col in row.index:
            level = row[level_col]
            confidence = row[conf_col]

            if not pd.isna(level) and not pd.isna(confidence):
                levels.append(level)
                confidences.append(confidence)

    if not levels:
        return None  # No valid values to determine a level

    # Find the maximum confidence
    max_confidence = max(confidences)
    
    # If there's only one max confidence, return the corresponding level
    max_indices = [i for i, c in enumerate(confidences) if c == max_confidence]
    if len(max_indices) == 1:
        return levels[max_indices[0]]

    # If multiple levels have the same confidence of 1, take the rounded average
    if max_confidence == 1:
        return round(sum(levels) / len(levels))

    # Otherwise, if there are ties but the confidence isn't 1, return the first level with max confidence
    return levels[max_indices[0]]

# Apply the function to create the 'Final Regularity Level' column
data['Final Regularity Level'] = data.apply(determine_final_level, axis=1)

# Display the length of data before cleaning
print(f"Number of data points before cleaning: {len(data)}")

import os
import trimesh

def load_and_validate_obj_file(file_path):
    """
    Load and validate an OBJ file to check for integrity.
    Returns a mesh if the file is valid, otherwise returns None.
    Performs checks for file existence, valid mesh object, 
    non-empty vertices, faces, surface area, and volume.
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    try:
        # Load the mesh
        mesh = trimesh.load(file_path)

        # Validate if loaded object is a Trimesh instance (i.e., a valid mesh)
        if not isinstance(mesh, trimesh.Trimesh):
            print(f"Skipping file {file_path}: Not a valid Mesh object.")
            return None

        # Check if the mesh has vertices
        if len(mesh.vertices) == 0:
            print(f"Skipping file {file_path}: No vertices found.")
            return None

        # Check if the mesh has faces
        if len(mesh.faces) == 0:
            print(f"Skipping file {file_path}: No faces found.")
            return None

        # Check if the mesh has non-zero surface area
        if mesh.area <= 0:
            print(f"Skipping file {file_path}: Surface area is zero or negative.")
            return None

        # Check if the mesh has non-zero volume (helps ensure the mesh is closed)
        if mesh.volume <= 0:
            print(f"Skipping file {file_path}: Volume is zero or negative.")
            return None

        # If all checks pass, return the mesh
        return mesh

    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None
    
def process_obj_files(base_dir, data, augment=False):
    """
    Process all OBJ files, validate them, and apply augmentations if specified.
    """
    validated_labels = []

    for index, row in data.iterrows():
        layer_folder = str(row['Object ID (Dataset Original Object ID)']).strip()
        obj_filename = layer_folder
        obj_file_path = os.path.join(base_dir, f"{layer_folder}", f"{obj_filename}.obj")

        vertices = load_and_validate_obj_file(obj_file_path)
        if vertices is not None:
            validated_labels.append(row)

    # Create a new DataFrame with validated labels only
    validated_labels_df = pd.DataFrame(validated_labels)
    validated_labels_df.reset_index(drop=True, inplace=True)
    validated_labels_df['Count No.'] = validated_labels_df.index + 1
    return validated_labels_df

def save_validated_data(validated_labels_df, output_file_path='datasets/abo/label/Final_Validated_Regularity_Levels.xlsx'):
    """
    Save the validated labels to an Excel file.
    """
    validated_labels_df.to_excel(output_file_path, index=False)
    print(f"Validated data saved to {output_file_path}")

# Main script execution
if __name__ == "__main__":
    base_dir = 'datasets/abo/obj-ABO'

    # Process the dataset to validate OBJ files
    validated_labels_df = process_obj_files(base_dir, data, augment=True)

    # Save the validated labels to the final Excel file
    save_validated_data(validated_labels_df, 'datasets/abo/label/Final_Validated_Regularity_Levels.xlsx')