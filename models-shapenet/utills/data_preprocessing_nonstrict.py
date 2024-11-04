import os
import pandas as pd
import numpy as np
import trimesh

# Load the dataset
input_excel_path = 'datasets/ShapeNetCoreV2/label/ShapeNetCoreV2_update.xlsx'
output_excel_path = 'datasets/ShapeNetCoreV2/label/Final_Regularized_labels.xlsx'

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

# Load and validate OBJ file with relaxed criteria
def load_and_validate_obj_file(file_path):
    if not os.path.exists(file_path):
        return None

    try:
        mesh = trimesh.load(file_path)
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return None
        return mesh
    except Exception:
        return None

# Process OBJ files with logging
def process_obj_files(base_dir, data):
    validated_labels = []
    skipped_count = 0  # To count the number of files skipped
    for index, row in data.iterrows():
        first_layer_folder = str(int(row['Folder Name'])).zfill(8)
        second_layer_folder = str(row['Object ID (Dataset Original Object ID)']).strip()
        obj_filename = 'model_normalized'
        obj_file_path = os.path.join(base_dir, 'obj-ShapeNetCoreV2', first_layer_folder, second_layer_folder, 'models', f"{obj_filename}.obj")

        mesh = load_and_validate_obj_file(obj_file_path)
        if mesh is not None:
            validated_labels.append(row)
        else:
            skipped_count += 1

        # Log progress every 500 files
        if (index + 1) % 500 == 0:
            print(f"Processed {index + 1} files, retained {len(validated_labels)}, skipped {skipped_count}.")

    validated_labels_df = pd.DataFrame(validated_labels)
    validated_labels_df.reset_index(drop=True, inplace=True)
    validated_labels_df['Count No.'] = validated_labels_df.index + 1
    print(f"Final count: {len(validated_labels)} retained, {skipped_count} skipped.")
    return validated_labels_df

# Save validated data
def save_validated_data(validated_labels_df, output_file_path):
    validated_labels_df.to_excel(output_file_path, index=False)
    print(f"Validated data saved to {output_file_path}")

# Main script execution
if __name__ == "__main__":
    labels_df = pd.read_excel(input_excel_path)

    # Determine the final regularity level for each row
    labels_df['Final Regularity Level'] = labels_df.apply(determine_final_level, axis=1)

    # Drop rows where no one labeled the mesh or the level is not in [1, 2, 3, 4]
    print("Filtering rows with valid regularity levels...")
    final_df = labels_df.dropna(subset=['Final Regularity Level'])
    final_df = final_df[final_df['Final Regularity Level'].isin([1, 2, 3, 4])]

    # Validate OBJ files with relaxed criteria
    print("Validating OBJ files...")
    # validated_labels_df = process_obj_files(base_dir='datasets/ShapeNetCoreV2', data=final_df)

    # Save the final dataset to an Excel file
    # save_validated_data(validated_labels_df, output_excel_path)
    save_validated_data(final_df, output_excel_path)
    # print(f"Input labels count: {len(labels_df)}, Final labels count: {len(validated_labels_df)}")
    print(f"Input labels count: {len(labels_df)}, Final labels count: {len(final_df)}")
