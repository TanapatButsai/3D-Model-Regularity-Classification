import os
import torch
import trimesh
from torch.utils.data import Dataset

class MeshDataset(Dataset):
    def __init__(self, base_dir, labels_df):
        self.base_dir = base_dir
        self.labels_df = labels_df

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        obj_id = self.labels_df.iloc[idx]['Object ID (Dataset Original Object ID)']
        label = self.labels_df.iloc[idx]['Final Regularity Level']

        obj_file = os.path.join(self.base_dir, obj_id, 'normalized_model.obj')

        # Check if the file exists and load it
        if not os.path.exists(obj_file):
            print(f"File not found: {obj_file}")
            return None, None

        mesh = trimesh.load(obj_file, force='mesh')

        # Check if mesh has vertices
        if mesh.vertices.size == 0:
            print(f"Skipping file {obj_file}: No vertices found.")
            return None, None

        points = mesh.sample(2048)  # Sample 2048 points from the mesh
        points = torch.tensor(points, dtype=torch.float32)

        return points, torch.tensor(label, dtype=torch.long)
