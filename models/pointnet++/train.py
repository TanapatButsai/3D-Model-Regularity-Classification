import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import PointNet2  # Assuming PointNet++ model
from data_loader import MeshDataset  # Custom data loader

# Configuration parameters
labels_file = 'datasets/3d-future-dataset/label/Final_Validated_Regularity_Levels.xlsx'
base_dir = 'datasets/3d-future-dataset/3D-FUTURE-model'
max_data_points = 128  # Adjust based on your dataset's typical size
batch_size = 16
num_epochs = 20
learning_rate = 0.001

# Initialize dataset and data loaders
train_dataset = MeshDataset(labels_file=labels_file, base_dir=base_dir, max_data_points=max_data_points)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, loss function, and optimizer
model = PointNet2(num_classes=4)  # Adjust num_classes as per your specific use case
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Check if CUDA is available and use GPU if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Training loop
print("Starting Training...")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for inputs, labels in train_loader:
        # Ensure input size is correct before training
        if inputs.shape[1] != max_data_points:
            print(f"Skipping batch with input shape: {inputs.shape}")
            continue

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Display the training progress
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}')

print("Training Complete!")