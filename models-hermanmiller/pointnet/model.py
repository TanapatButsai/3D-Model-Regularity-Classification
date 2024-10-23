import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNet(nn.Module):
    def __init__(self, num_classes=5):  # Adjust the number of classes to match the "Final Regularity Level"
        super(PointNet, self).__init__()
        
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
    def forward(self, x):
        # Input x shape: (batch_size, num_points, 3)
        x = x.permute(0, 2, 1)  # Change to (batch_size, 3, num_points)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = torch.max(x, 2, keepdim=True)[0]  # Perform max pooling
        x = x.view(-1, 1024)  # Flatten for fully connected layers
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)  # Output layer for classification
        
        return F.log_softmax(x, dim=1)

# Example usage
if __name__ == "__main__":
    num_points = 1024
    num_classes = 5  # Number of classes based on "Final Regularity Level"
    model = PointNet(num_classes=num_classes)
    
    # Create random input point cloud with shape (batch_size, num_points, 3)
    batch_size = 8
    points = torch.randn(batch_size, num_points, 3)
    
    # Forward pass
    outputs = model(points)
    print(outputs.shape)  # Should be (batch_size, num_classes)
