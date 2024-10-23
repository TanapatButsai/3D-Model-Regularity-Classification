import os
import pandas as pd

# Specify the path to the folder where you want to count subfolders
base_folder = 'datasets/3d-future-dataset/objs'  # Replace with your folder path

# Count the number of subfolders in the specified folder
folder_count = len([entry for entry in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, entry))])

# Display the result
print(f'The number of folders in "{base_folder}" is: {folder_count}')

# Configuration
ori_file_path = 'datasets/3d-future-dataset/label/3D-FUTURE-Layout.xlsx'  # Replace with your file path
final_path = 'datasets/3d-future-dataset/label/Final_Validated_Regularity_Levels.xlsx'

# Step 1: Load and Clean the Labels Data
ori_labels_df = pd.read_excel(ori_file_path)
final_labels_df = pd.read_excel(final_path)

print(f"Initial number of data points: {len(ori_labels_df)}")
print(f"After number of data points: {len(final_labels_df)}")
print(f"Initial - After = {len(ori_labels_df) - len(final_labels_df)}")

# final_labels_df['Final Regularity Level'] = final_labels_df['Final Regularity Level'] - 1
print("Unique labels in the dataset:", final_labels_df['Final Regularity Level'].unique())