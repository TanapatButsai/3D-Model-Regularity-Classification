import os
import pandas as pd
from pytorch3d.io import load_objs_as_meshes

# Configuration
file_path = 'datasets/3d-future-dataset/label/3D-FUTURE-Layout.xlsx'  # Replace with your file path
output_file_path = 'datasets/3d-future-dataset/label/Final_Validated_Regularity_Levels.xlsx'

base_dir = 'datasets/3d-future-dataset/3D-FUTURE-model'  # Set the base directory for your 3D models

# Step 1: Load and Clean the Labels Data
labels_df = pd.read_excel(file_path)
print(f"Initial number of data points: {len(labels_df)}")

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

# Step 2: Remove rows with NaN values and labels equal to 0
labels_df = labels_df.dropna(subset=['Final Regularity Level'])
labels_df['Final Regularity Level'] = labels_df['Final Regularity Level'].astype(int)
labels_df = labels_df[labels_df['Final Regularity Level'] > 0]
print(f"Number of data points after label cleaning: {len(labels_df)}")

# Step 3: Validate the OBJ files
valid_rows = []

for index, row in labels_df.iterrows():
    obj_id = row['Object ID (Dataset Original Object ID)']
    obj_file = os.path.join(base_dir, obj_id, 'normalized_model.obj')

    # Check if the file exists and can be loaded without errors
    if os.path.exists(obj_file):
        try:
            mesh = load_objs_as_meshes([obj_file])
            valid_rows.append(row)
        except Exception as e:
            print(f"File '{obj_file}' failed to load due to: {e}")
    else:
        print(f"File not found: {obj_file}")

# Step 4: Create a DataFrame with the valid rows only
validated_df = pd.DataFrame(valid_rows)

# Step 5: Save the cleaned and validated labels to a new Excel file
validated_df.to_excel(output_file_path, index=False)
print(f"Final number of valid data points: {len(validated_df)}")
print(f"Validated data saved to: {output_file_path}")
