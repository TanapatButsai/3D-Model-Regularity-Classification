import torch
from torch.utils.data import Dataset
import numpy as np
import os
import trimesh

class MeshDataset(Dataset):
    def __init__(self, base_dir, labels_df, transform=None):
        self.base_dir = base_dir
        self.labels_df = labels_df
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        obj_id = self.labels_df.iloc[idx]['Object ID (Dataset Original Object ID)']
        label = self.labels_df.iloc[idx]['Final Regularity Level']

        obj_file = os.path.join(self.base_dir, obj_id, 'normalized_model.obj')

        if not os.path.exists(obj_file):
            print(f"File not found: {obj_file}")
            return None, None

        mesh = trimesh.load(obj_file, force='mesh')
        points = mesh.sample(2048)  # Sample points from the mesh
        points = torch.tensor(points, dtype=torch.float32)

        if self.transform:
            points = self.transform(points)

        return points, torch.tensor(label, dtype=torch.long)