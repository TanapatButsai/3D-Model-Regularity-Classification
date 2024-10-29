import os
import pandas as pd
import numpy as np
import trimesh

# Load the dataset
input_excel_path = 'datasets\ShapeNetCoreV2\label\ShapeNetCoreV2_update.xlsx'  # Path to the input Excel file
output_excel_path = 'datasets/ShapeNetCoreV2/label/Final_Regularized_labels.xlsx'  # Path to the output Excel file

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
            if not pd.isna(row[level]) and row[level] in [1, 2, 3, 4]:
                return row[level]
    return np.nan

# Load and validate OBJ file
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

# Process OBJ files
def process_obj_files(base_dir, data, augment=False):
    """
    Process all OBJ files, validate them, and apply augmentations if specified.
    """
    validated_labels = []

    for index, row in data.iterrows():
        first_layer_folder = str(int(row['Folder Name'])).zfill(8)
        second_layer_folder = str(row['Object ID (Dataset Original Object ID)']).strip()
        obj_filename = 'model_normalized'
        obj_file_path = os.path.join(base_dir, 'obj-ShapeNetCoreV2', first_layer_folder, second_layer_folder, 'models', f"{obj_filename}.obj")

        mesh = load_and_validate_obj_file(obj_file_path)
        if mesh is not None:
            validated_labels.append(row)

    # Create a new DataFrame with validated labels only
    validated_labels_df = pd.DataFrame(validated_labels)
    validated_labels_df.reset_index(drop=True, inplace=True)
    validated_labels_df['Count No.'] = validated_labels_df.index + 1
    return validated_labels_df

# Save validated data
def save_validated_data(validated_labels_df, output_file_path='datasets/pix3d/label/Final_Validated_Regularity_Levels.xlsx'):
    """
    Save the validated labels to an Excel file.
    """
    validated_labels_df.to_excel(output_file_path, index=False)
    print(f"Validated data saved to {output_file_path}")

# Main script execution
if __name__ == "__main__":
    # Load the dataset
    input_excel_path = 'datasets/ShapeNetCoreV2/label/ShapeNetCoreV2_update.xlsx'  # Path to the input Excel file
    output_excel_path = 'datasets/ShapeNetCoreV2/label/Final_Validated_Regularity_Levels.xlsx'  # Path to the output Excel file

    labels_df = pd.read_excel(input_excel_path)

    # Determine the final regularity level for each row
    labels_df['Final Regularity Level'] = labels_df.apply(determine_final_level, axis=1)

    # Drop rows where no one labeled the mesh or the level is not in [1, 2, 3, 4]
    final_df = labels_df.dropna(subset=['Final Regularity Level'])
    final_df = final_df[final_df['Final Regularity Level'].isin([1, 2, 3, 4])]

    # Validate OBJ files
    validated_labels_df = process_obj_files(base_dir='datasets/ShapeNetCoreV2', data=final_df)

    # Save the final dataset to an Excel file
    save_validated_data(validated_labels_df, output_excel_path)

    print(f"Input labels count: {len(labels_df)}, Final labels count: {len(validated_labels_df)}")