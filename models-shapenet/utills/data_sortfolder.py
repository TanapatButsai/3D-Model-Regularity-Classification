import openpyxl
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define paths
list_dir_path = Path("datasets/ShapeNetCoreV2/obj-ShapeNetCoreV2/List_all_name")
excel_path = Path("datasets/ShapeNetCoreV2/label/ShapeNetCoreV2.xlsx")
updated_excel_path = Path("datasets/ShapeNetCoreV2/label/ShapeNetCoreV2_update.xlsx")

# Check if directory exists
if not list_dir_path.exists():
    raise FileNotFoundError(f"Directory not found: {list_dir_path}")

if not excel_path.exists():
    raise FileNotFoundError(f"Excel file not found: {excel_path}")

# Load the Excel workbook and sheet
wb = openpyxl.load_workbook(excel_path)
sheet = wb.active

# Iterate through all text files in the directory
for list_file_path in list_dir_path.glob("lists_*.txt"):
    # Extract the prefix number from the filename
    prefix = list_file_path.stem.split("_")[1]

    # Load the list of folder names from the text file
    with open(list_file_path, "r") as f:
        folder_ids = {line.strip() for line in f}
    logging.info(f"Loaded {len(folder_ids)} folder IDs from {list_file_path}")

    # Iterate through column B and match folder names
    for row in range(2, sheet.max_row + 1):
        object_id = sheet[f"B{row}"].value
        if object_id in folder_ids:
            # Write the folder name in column C with the prefix
            sheet[f"C{row}"] = str(prefix)
            logging.info(f"Updated row {row} with folder ID '{prefix}'")

# Save the modified workbook
wb.save(updated_excel_path)
logging.info(f"Saved updated Excel file to {updated_excel_path}")