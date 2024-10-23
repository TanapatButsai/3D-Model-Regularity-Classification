import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pytorch3d.io import load_objs_as_meshes

class MeshDataset(Dataset):
    def __init__(self, base_dir, labels_df, num_points=1024, augment=False):
        self.base_dir = base_dir
        self.labels_df = labels_df
        self.num_points = num_points
        self.augment = augment

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        obj_id = self.labels_df.iloc[idx]['Object ID (Dataset Original Object ID)']
        label = self.labels_df.iloc[idx]['Final Regularity Level'] - 1  # Adjust labels to be 0-indexed

        obj_file = os.path.join(self.base_dir, obj_id, 'normalized_model.obj')

        # Check if the file exists
        if not os.path.exists(obj_file):
            print(f"File not found: {obj_file}")
            return None, None  # Skip the sample if the file does not exist

        point_cloud = self.mesh_to_point_cloud(obj_file)

        if point_cloud is None:
            return None, None  # Skip the sample if mesh processing fails

        # Apply data augmentation if the augment flag is set to True
        if self.augment:
            point_cloud = self.augment_point_cloud(point_cloud)

        return torch.tensor(point_cloud.T, dtype=torch.float32), label

    def mesh_to_point_cloud(self, obj_file):
        try:
            mesh = load_objs_as_meshes([obj_file])
            vertices = mesh.verts_packed().numpy()

            # Randomly sample points from the mesh vertices to create a point cloud
            point_indices = np.random.choice(vertices.shape[0], self.num_points, replace=True)
            point_cloud = vertices[point_indices]

            return point_cloud
        except Exception as e:
            print(f"Error processing file {obj_file}: {e}")
            return None

    def augment_point_cloud(self, points):
        # Apply random rotations to the point cloud
        angle = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                    [np.sin(angle), np.cos(angle), 0],
                                    [0, 0, 1]])
        points = np.dot(points, rotation_matrix)

        # Add Gaussian noise to the point cloud
        noise = np.random.normal(0, 0.02, points.shape)
        points += noise

        # Randomly scale the point cloud
        scale = np.random.uniform(0.9, 1.1)
        points *= scale

        return points
