import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Paths
excel_path = 'datasets/3d-future-dataset/label/Final_Validated_Regularity_Levels.xlsx'
base_dir = 'datasets/3d-future-dataset/obj-3d.future'
output_dir = 'dataset'  # Final dataset directory

# Load Excel file
labels_df = pd.read_excel(excel_path)

# Ensure Final Regularity Level is an integer
labels_df['Final Regularity Level'] = labels_df['Final Regularity Level'].astype(int)

# Filter out OBJ IDs with invalid levels
labels_df = labels_df[labels_df['Final Regularity Level'] > 0]

# Split into train/test (e.g., 80/20 split)
train_df, test_df = train_test_split(labels_df, test_size=0.2, random_state=42)

# Function to organize files
def organize_files(df, dataset_type):
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Organizing {dataset_type} data"):
        obj_id = row['Object ID (Dataset Original Object ID)']
        class_label = f"class{row['Final Regularity Level']}"
        source_file = os.path.join(base_dir, obj_id, 'normalized_model.obj')

        # Destination path
        dest_dir = os.path.join(output_dir, dataset_type, class_label)
        os.makedirs(dest_dir, exist_ok=True)

        dest_file = os.path.join(dest_dir, f"{obj_id}.obj")
        try:
            # Copy file to the destination
            shutil.copyfile(source_file, dest_file)
        except FileNotFoundError:
            print(f"File not found: {source_file}")
        except Exception as e:
            print(f"Error processing {obj_id}: {e}")

# Organize training data
organize_files(train_df, 'train')

# Organize testing data
organize_files(test_df, 'test')

print("Dataset organization complete.")
