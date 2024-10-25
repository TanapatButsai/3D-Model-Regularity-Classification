import pandas as pd
import numpy as np

# Load the dataset
input_excel_path = 'datasets/shapenetcore/label/shapenet.xlsx'  # Path to the input Excel file
output_excel_path = 'datasets/shapenetcore/label/final_regularized_labels.xlsx'  # Path to the output Excel file

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

# Main script execution
if __name__ == "__main__":
    labels_df = pd.read_excel(input_excel_path)

    # Determine the final regularity level for each row
    labels_df['Final Regularity Level'] = labels_df.apply(determine_final_level, axis=1)

    # Drop rows where no one labeled the mesh or the level is not in [1, 2, 3, 4]
    final_df = labels_df.dropna(subset=['Final Regularity Level'])
    final_df = final_df[final_df['Final Regularity Level'].isin([1, 2, 3, 4])]

    # Save the final dataset to an Excel file
    final_df.to_excel(output_excel_path, index=False)

    print(f"Input labels count: {len(labels_df)}, Final labels count: {len(final_df)}")
