import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from model import PointNet2
from data_loader import MeshDataset

# Configuration settings
MAX_DATA_POINTS = 10000  # Maximum number of data points for training
BATCH_SIZE = 32
NUM_CLASSES = 4
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 5

# Load the label data
file_path = 'data/label/Final_Regularity_Levels.xlsx'
labels_df = pd.read_excel(file_path)

# Display dataset size before cleaning
print(f"Number of data points before cleaning: {len(labels_df)}")

# Clean the dataset: remove rows with NaN values and labels equal to 0
labels_df = labels_df.dropna(subset=['Final Regularity Level'])
labels_df['Final Regularity Level'] = labels_df['Final Regularity Level'].astype(int)
labels_df = labels_df[labels_df['Final Regularity Level'] > 0]

# Limit the data points to the specified maximum and randomly shuffle the data
labels_df = labels_df.sample(n=MAX_DATA_POINTS, random_state=42)
print(f"Number of data points after cleaning and limiting: {len(labels_df)}")

# Split the data into training and validation sets
train_df, val_df = train_test_split(labels_df, test_size=0.2, random_state=42)

# Set the base directory for your 3D models
base_dir = 'data/3D-FUTURE-model'

# Create the dataset and data loaders
train_dataset = MeshDataset(base_dir=base_dir, labels_df=train_df)
val_dataset = MeshDataset(base_dir=base_dir, labels_df=val_df)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize the model, optimizer, and loss function
model = PointNet2(num_classes=NUM_CLASSES)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Training loop with validation
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0
    for inputs, labels in train_loader:
        if inputs is None:  # Skip invalid data
            continue
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}')

    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            if inputs is None:  # Skip invalid data
                continue
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Validation Loss: {avg_val_loss:.4f}')

    # Check for improvement in validation loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        print("Early stopping triggered.")
        break

print("Training complete!")
