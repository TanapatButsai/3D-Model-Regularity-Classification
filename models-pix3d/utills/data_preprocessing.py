import os
import pandas as pd
import numpy as np
import trimesh
from tqdm import tqdm

def determine_final_level(row):
    try:
        # List persons and their layout level columns and confidence columns
        persons = ['Person 1', 'Person 2', 'Person 3', 'Person 4']
        levels = []
        confidences = []

        # Extract layout levels and confidences
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
            return None  # No valid levels

        # Use the level with the highest confidence
        max_confidence = max(confidences)
        max_indices = [i for i, c in enumerate(confidences) if c == max_confidence]

        if len(max_indices) == 1:
            return levels[max_indices[0]]
        
        # If multiple levels share max confidence of 1, take average
        if max_confidence == 1:
            return round(sum(levels) / len(levels))

        # Otherwise, return the first level with max confidence
        return levels[max_indices[0]]

    except (ValueError, TypeError):
        return None

def load_and_validate_obj_file(file_path):
    """
    Load and validate an OBJ file to check for integrity.
    Returns vertices if the file is valid, otherwise returns None.
    """
    if not os.path.exists(file_path):
        return None, f"File not found: {file_path}"
    
    try:
        mesh = trimesh.load(file_path)
        if isinstance(mesh, trimesh.Scene) and mesh.geometry:
            mesh = list(mesh.geometry.values())[0]
        
        vertices = mesh.vertices
        if len(vertices) == 0:
            return None, f"Skipping file {file_path}: No vertices found."
        return vertices, None
    except Exception as e:
        return None, f"Error loading file {file_path}: {e}"

def process_obj_files(base_dir, data):
    """
    Process all OBJ files and validate them.
    """
    validated_labels = []
    error_messages = []

    for index, row in tqdm(data.iterrows(), total=data.shape[0], desc="Processing OBJ files"):
        layer_folder = str(row['Object ID (Dataset Original Object ID)']).strip()
        obj_file_path = os.path.join(base_dir, 'models', layer_folder.strip(), "model.obj")

        vertices, error_message = load_and_validate_obj_file(obj_file_path)
        if vertices is not None:
            validated_labels.append(row)
        else:
            if error_message:
                error_messages.append(error_message)

    # Print error messages after processing
    for message in error_messages:
        tqdm.write(message)

    # Create a DataFrame with validated labels
    validated_labels_df = pd.DataFrame(validated_labels)
    validated_labels_df.reset_index(drop=True, inplace=True)
    validated_labels_df['Count No.'] = validated_labels_df.index + 1
    return validated_labels_df

def check_mesh_level_classification(base_dir, excel_path):
    # Load Excel data with mesh information
    try:
        data = pd.read_excel(excel_path)
    except FileNotFoundError:
        print(f"Error: The file '{excel_path}' does not exist.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

    # Apply final layout level determination
    data['Final Layout Level'] = data.apply(determine_final_level, axis=1)
    data = data.dropna(subset=['Final Layout Level'])
    data['Final Layout Level'] = data['Final Layout Level'].astype(int)
    data = data[data['Final Layout Level'] > 0]

    # Process OBJ files
    validated_labels_df = process_obj_files(base_dir, data)

    # Save validated labels to Excel
    output_file_path = 'datasets/pix3d/label/Final_Validated_Regularity_Levels.xlsx'
    validated_labels_df.to_excel(output_file_path, index=False)
    print(f"Validated data saved to {output_file_path}")

    # Summary
    total_files = data.shape[0]
    validated_count = len(validated_labels_df)
    failed_count = total_files - validated_count
    print("\nSummary:")
    print(f"Total files processed: {total_files}")
    print(f"Successfully validated files: {validated_count}")
    print(f"Failed validations: {failed_count}")

    return validated_labels_df

# Example usage
if __name__ == "__main__":
    base_dir = "datasets/pix3d/obj-pix3d"
    excel_path = 'datasets/pix3d/label/pix3d.xlsx'
    check_mesh_level_classification(base_dir, excel_path)