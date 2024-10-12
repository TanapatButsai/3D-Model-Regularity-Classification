import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import trimesh
import numpy as np

class MeshDataset(Dataset):
    def __init__(self, labels_file, base_dir, max_data_points):
        self.labels_df = pd.read_excel(labels_file).head(max_data_points)
        self.base_dir = base_dir
        self.max_vertices = 50000  # Set a maximum number of vertices for padding/truncation

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        label_info = self.labels_df.iloc[idx]
        obj_id = label_info['Object ID (Dataset Original Object ID)']
        obj_file_path = os.path.join(self.base_dir, obj_id, 'normalized_model.obj')

        vertices = self.load_vertices(obj_file_path)
        label = int(label_info['Final Regularity Level'])

        return torch.tensor(vertices, dtype=torch.float32).permute(1, 0), torch.tensor(label, dtype=torch.long)

    def load_vertices(self, file_path):
        """
        Load vertices from an OBJ file and ensure consistent size.
        """
        try:
            mesh = trimesh.load(file_path)
            vertices = mesh.vertices
            if len(vertices) == 0:
                raise ValueError("No vertices found in OBJ file.")
            # Pad or truncate vertices to ensure consistent size
            if len(vertices) > self.max_vertices:
                vertices = vertices[:self.max_vertices]
            else:
                padding = self.max_vertices - len(vertices)
                vertices = np.pad(vertices, ((0, padding), (0, 0)), mode='constant', constant_values=0)
            return vertices
        except Exception as e:
            print(f"Error loading vertices from {file_path}: {e}")
            return np.zeros((self.max_vertices, 3))  # Return a zero tensor if loading fails