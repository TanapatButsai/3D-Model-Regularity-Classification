import os
import pandas as pd
import numpy as np
import trimesh

# Load the raw label data
excel_path = 'datasets\ikea\label\ikea.xlsx'
data = pd.read_excel(excel_path)

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
data['Final Regularity Level'] = data.apply(determine_final_level, axis=1)

# Remove rows with NaN values in 'Final Regularity Level'
data = data.dropna(subset=['Final Regularity Level'])

# Convert 'Final Regularity Level' to integer and remove rows where the level is 0
data['Final Regularity Level'] = data['Final Regularity Level'].astype(int)
data = data[data['Final Regularity Level'] > 0]

# Display the length of data before cleaning
print(f"Number of data points before cleaning: {len(data)}")

def load_and_validate_obj_file(file_path):
    """
    Load and validate an OBJ file to check for integrity.
    Returns vertices if the file is valid, otherwise returns None.
    Also checks if the file exists for 3D model classification.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    try:
        mesh = trimesh.load(file_path)
        # Only accept if the loaded object is a Mesh
        if not isinstance(mesh, trimesh.Trimesh):
            print(f"Skipping file {file_path}: Not a valid Mesh object.")
            return None
        
        vertices = mesh.vertices
        if len(vertices) == 0:
            print(f"Skipping file {file_path}: No vertices found.")
            return None
        return vertices
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def process_obj_files(base_dir, data, augment=False):
    """
    Process all OBJ files, validate them, and apply augmentations if specified.
    """
    validated_labels = []

    for index, row in data.iterrows():
        first_layer_folder = str(row['FolderName']).strip()
        second_layer_folder = str(row['Object ID (Dataset Original Object ID)']).strip()
        obj_filename = 'ikea_model'
        obj_file_path = os.path.join(base_dir, first_layer_folder, f"{second_layer_folder}", f"{obj_filename}.obj")

        vertices = load_and_validate_obj_file(obj_file_path)
        if vertices is not None:
            validated_labels.append(row)

    # Create a new DataFrame with validated labels only
    validated_labels_df = pd.DataFrame(validated_labels)
    validated_labels_df.reset_index(drop=True, inplace=True)
    validated_labels_df['Count No.'] = validated_labels_df.index + 1
    return validated_labels_df

def save_validated_data(validated_labels_df, output_file_path='datasets/ikea/label/Final_Validated_Regularity_Levels.xlsx'):
    """
    Save the validated labels to an Excel file.
    """
    validated_labels_df.to_excel(output_file_path, index=False)
    print(f"Validated data saved to {output_file_path}")

# Main script execution
if __name__ == "__main__":
    base_dir = 'datasets/ikea/obj-IKEA'

    # Process the dataset to validate OBJ files
    validated_labels_df = process_obj_files(base_dir, data, augment=True)

    # Save the validated labels to the final Excel file
    save_validated_data(validated_labels_df, 'datasets/ikea/label/Final_Validated_Regularity_Levels.xlsx')