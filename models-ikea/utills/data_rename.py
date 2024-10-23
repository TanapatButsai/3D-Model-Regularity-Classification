import os
import pandas as pd

# Load the raw label data
excel_path = 'datasets/ikea/label/ikea.xlsx'
data = pd.read_excel(excel_path)

# Print column names to verify their correctness
print("Column names:", data.columns)

# Function to determine the final path
def get_target_folder(base_dir, folder_name, object_id):
    folder_name = folder_name.strip()
    object_id = object_id.strip()
    path_parts = [base_dir, folder_name, object_id]
    return os.path.normpath(os.path.join(*path_parts))

# Base directory where the folders are located
base_dir = 'datasets/ikea/obj-IKEA'  # Replace with the actual base directory

# Extract folder data from the Excel file
folders = list(zip(data['FolderName'], data['Object ID (Dataset Original Object ID)']))

# Loop through the folder data
for folder_name, object_id in folders:
    # Construct the full path to the target folder using the function
    target_folder = get_target_folder(base_dir, folder_name, object_id)

    # Check if the folder exists
    if os.path.exists(target_folder):
        print(f'Processing folder: {target_folder}')
        # Get all ".obj" files in the folder
        for file_name in os.listdir(target_folder):
            if file_name.endswith(".obj"):
                # Construct the full path to the original and new file names
                original_file_path = os.path.join(target_folder, file_name)
                new_file_path = os.path.join(target_folder, "ikea_model.obj")
                
                # Rename the file
                os.rename(original_file_path, new_file_path)
                print(f'Renamed: {original_file_path} to {new_file_path}')
    else:
        print(f'Folder not found: {target_folder}')