### model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        last_channel = in_channel
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        # For simplicity, we'll skip the sampling and grouping steps
        new_xyz = xyz[:, :self.npoint, :]  # Placeholder for the new centroid positions
        new_points = torch.cat([xyz, points], dim=-1) if points is not None else xyz  # Placeholder
        new_points = new_points.permute(0, 2, 1).unsqueeze(-1)  # Adjust shape for Conv2d
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        new_points = new_points.squeeze(-1).permute(0, 2, 1)  # Adjust shape back
        return new_xyz, new_points

class PointNet2(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2, self).__init__()
        self.sa1 = PointNetSetAbstraction(512, 0.2, 32, 3, [64, 64, 128])
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 128 + 3, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(None, None, None, 256 + 3, [256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        print(f"Input to PointNet2: {x.shape}")
        B, _, _, _ = x.shape  # Expecting shape [batch_size, num_channels, num_points, 1]
        x = x.squeeze(-1)  # Remove the last dimension to get [batch_size, num_channels, num_points]
        l1_xyz, l1_points = self.sa1(x, None)
        print(f"After SA1 - l1_xyz: {l1_xyz.shape}, l1_points: {l1_points.shape}")
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        print(f"After SA2 - l2_xyz: {l2_xyz.shape}, l2_points: {l2_points.shape}")
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        print(f"After SA3 - l3_xyz: {l3_xyz.shape}, l3_points: {l3_points.shape}")
        x = l3_points.view(B, -1)  # Flatten the feature vector
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x