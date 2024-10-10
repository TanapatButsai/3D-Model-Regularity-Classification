import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from model import PointNet2
from data_loader import MeshDataset

# Configuration settings
MAX_DATA_POINTS = 10000
BATCH_SIZE = 32
NUM_CLASSES = 4
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 5

# Load the label data
file_path = 'datasets/3d-future-dataset/label/Final_Validated_Regularity_Levels.xlsx'
labels_df = pd.read_excel(file_path)

# Clean and shuffle the data
labels_df = labels_df.dropna(subset=['Final Regularity Level'])
labels_df['Final Regularity Level'] = labels_df['Final Regularity Level'].astype(int)
labels_df = labels_df.sample(n=MAX_DATA_POINTS, random_state=42)

# Train-validation split
train_df, val_df = train_test_split(labels_df, test_size=0.2, random_state=42)

# Create datasets and data loaders
base_dir = 'datasets/3d-future-dataset/3D-FUTURE-model'
train_dataset = MeshDataset(base_dir, train_df)
val_dataset = MeshDataset(base_dir, val_df)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize model, optimizer, and loss function
model = PointNet2(num_classes=NUM_CLASSES)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

best_val_loss = float('inf')
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0
    for inputs, labels in train_loader:
        if inputs is None:
            continue

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            if inputs is None:
                continue

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_pointnet2_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        print("Early stopping triggered.")
        break
