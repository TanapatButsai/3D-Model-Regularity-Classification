
import pandas as pd

# Data for the table
data = {
    "Dataset": ["3D-FUTURE", "3D-FUTURE", "ABO", "ABO", "HermanMiller", "HermanMiller",
                "IKEA", "IKEA", "Pix3D", "Pix3D", "ShapeNetCore", "ShapeNetCore"],
    "Model": ["PointNet", "PointNet++", "PointNet", "PointNet++", "PointNet", "PointNet++",
              "PointNet", "PointNet++", "PointNet", "PointNet++", "PointNet", "PointNet++"],
    "Epochs": [100, 100, 80, 100, 60, 100, 10, 100, 50, 50, 50, 50],
    "Learning Rate": [0.001, 0.0005, 0.0005, 0.0005, 0.001, 0.0005, 0.001, 0.0005, 0.001, 0.001, 0.001, 0.001],
    "Batch Size": [32, 32, 32, 32, 16, 32, 4, 32, 16, 16, 32, 32],
    "Weight Decay": ["N/A", 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, "N/A", 1e-5, "N/A", "N/A", "N/A", "N/A"],
    "Dropout Rate": ["N/A", "N/A", 0.4, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"],
    "Patience (Early Stopping)": [10, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", 5, 5],
    "Class Weights": ["N/A", "N/A", "N/A", "N/A", [1.5, 1.5, 1.8, 1.0], "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to Excel
file_path = "Step3_Parameter_Tuning_config.xlsx"
df.to_excel(file_path, index=False)

file_path