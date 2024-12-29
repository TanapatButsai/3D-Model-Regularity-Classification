import pandas as pd

# Data for the table including MeshNet parameters
data = {
    "Dataset": [
        "3D-FUTURE", "3D-FUTURE", "3D-FUTURE",
        "ABO", "ABO", "ABO",
        "HermanMiller", "HermanMiller", "HermanMiller",
        "IKEA", "IKEA", "IKEA",
        "Pix3D", "Pix3D", "Pix3D",
        "ShapeNetCore", "ShapeNetCore", "ShapeNetCore"
    ],
    "Model": [
        "PointNet", "PointNet++", "MeshNet",
        "PointNet", "PointNet++", "MeshNet",
        "PointNet", "PointNet++", "MeshNet",
        "PointNet", "PointNet++", "MeshNet",
        "PointNet", "PointNet++", "MeshNet",
        "PointNet", "PointNet++", "MeshNet"
    ],
    "Epochs": [
        100, 100, 70,
        80, 100, 70,
        60, 100, 70,
        10, 100, 70,
        50, 50, 60,
        50, 50, 70
    ],
    "Learning Rate": [
        0.001, 0.0005, 0.001,
        0.0005, 0.0005, 0.001,
        0.001, 0.0005, 0.001,
        0.001, 0.0005, 0.001,
        0.001, 0.001, 0.001,
        0.001, 0.001, 0.001
    ],
    "Batch Size": [
        32, 32, 32,
        32, 32, 32,
        16, 32, 16,
        4, 32, 32,
        16, 16, 16,
        32, 32, 32
    ],
    "Weight Decay": [
        "N/A", 1e-5, 1e-4,
        1e-5, 1e-5, 1e-4,
        1e-5, 1e-5, 1e-4,
        "N/A", 1e-5, 1e-4,
        "N/A", "N/A", 1e-4,
        "N/A", "N/A", 1e-4
    ],
    "Dropout Rate": [
        "N/A", "N/A", 0.3,
        0.4, "N/A", 0.3,
        "N/A", "N/A", 0.3,
        "N/A", "N/A", 0.3,
        "N/A", "N/A", 0.3,
        "N/A", "N/A", 0.3
    ],
    "Patience (Early Stopping)": [
        10, "N/A", 5,
        "N/A", "N/A", 5,
        "N/A", "N/A", 5,
        "N/A", "N/A", 5,
        "N/A", "N/A", 5,
        5, 5, 5
    ],
    "Class Weights": [
        "N/A", "N/A", "N/A",
        "N/A", "N/A", "N/A",
        [1.5, 1.5, 1.8, 1.0], "N/A", "N/A",
        "N/A", "N/A", "N/A",
        "N/A", "N/A", "N/A",
        "N/A", "N/A", "N/A"
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to Excel
file_path = "Parameter_Tuning_with_MeshNet.xlsx"
df.to_excel(file_path, index=False)

print(f"Excel file saved at: {file_path}")
