import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from data_loader import MeshDataset
from model import PointNet
import time

# Configuration
MAX_DATA_POINTS = 5000
num_epochs = 50
batch_size = 32
learning_rate = 0
num_classes = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Step 1: Load and Prepare the Labels Data
# DONT FORGET TO RUN "data_preprocessing.py" BEFORE TRAIN
file_path = 'datasets/3d-future-dataset/label/Final_Validated_Regularity_Levels.xlsx'  # Replace with your file path
labels_df = pd.read_excel(file_path)

# Step 2: Randomly sample data points from the cleaned dataset
labels_df = labels_df.sample(n=min(MAX_DATA_POINTS, len(labels_df)), random_state=42)  # random_state ensures reproducibility

# Step 3: Split the data into training and validation sets
train_size = int(0.8 * len(labels_df))
val_size = len(labels_df) - train_size
train_labels_df = labels_df.iloc[:train_size]
val_labels_df = labels_df.iloc[train_size:]

# Step 4: Create datasets and data loaders
base_dir = 'datasets/3d-future-dataset/3D-FUTURE-model'
train_dataset = MeshDataset(base_dir=base_dir, labels_df=train_labels_df, augment=True)
val_dataset = MeshDataset(base_dir=base_dir, labels_df=val_labels_df, augment=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Step 5: Initialize the model, loss function, and optimizer
model = PointNet(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop with early stopping
best_val_loss = float('inf')
patience = 5
patience_counter = 0
start_time = time.time()

print("start!")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for inputs, labels in train_loader:
        if inputs is None or labels is None or len(inputs) == 0 or len(labels) == 0:
            continue  # Skip invalid samples

        inputs, labels = inputs.to(device), labels.to(device).long()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation step
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            if inputs is None or labels is None or len(inputs) == 0 or len(labels) == 0:
                continue  # Skip invalid samples

            inputs, labels = inputs.to(device), labels.to(device).long()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')

    # Early stopping logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")

# Load the best model and evaluate
model.load_state_dict(torch.load('best_model.pth'))
print("Classification Report:")
print(classification_report(all_labels, all_preds, zero_division=0))
