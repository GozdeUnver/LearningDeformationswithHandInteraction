from pathlib import Path
import json

import numpy as np
import torch
import os
import cv2
import open3d
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,path=None,deformation_path=None,num_points=2048, split_vectors=False):
        super().__init__()
        self.input_paths = ["./data/pointcloud_sampled/YellowToy01/inputs/non_deformed_2_correspondences_zoom_2048.ply"]
        self.target_paths = ["./data/pointcloud_sampled/YellowToy01/targets/deformed_2_correspondences_zoom_2048.ply"]
        self.deformation_nondeformed=["./data/pointcloud_sampled/YellowToy01/deformations/non_deformed_2_correspondences_zoom_2048_paired_648.ply"]
        self.deformation_deformed=["./data/pointcloud_sampled/YellowToy01/deformations/deformed_2_correspondences_zoom_2048_paired_648.ply"]
        self.num_points=num_points
        self.target_num_all = 3
        self.target_start_index = 0
        self.split_vectors = split_vectors
            
    def __getitem__(self, index):
        index = 0 # THIS IS BECAUSE OF BATCHNORMS - CHANGE WHEN NOT OVERFITTING!
        input_mesh=np.asarray(open3d.io.read_point_cloud(self.input_paths[index]).points, dtype=np.float32)
        target_mesh=np.asarray(open3d.io.read_point_cloud(self.target_paths[index]).points, dtype=np.float32)
        deformed_mesh=np.asarray(open3d.io.read_point_cloud(self.deformation_deformed[index]).points, dtype=np.float32)
        non_deformed_mesh=np.asarray(open3d.io.read_point_cloud(self.deformation_nondeformed[index]).points, dtype=np.float32)
        deformation_vectors = non_deformed_mesh-deformed_mesh
        distances = np.linalg.norm(deformation_vectors, axis=1)
        max_index=np.argmax(distances)
        deformation_vector=deformation_vectors[max_index]
        mesh_index=np.where(input_mesh == non_deformed_mesh[max_index])
        deformations=np.zeros(input_mesh.shape)
        deformations[mesh_index,:] =deformation_vector
        if self.split_vectors:
            return (input_mesh, deformations), target_mesh
        else:
            input_mesh = np.concatenate([input_mesh, deformations], axis=1)
            return input_mesh, target_mesh
    
    def __len__(self):
        return 2 * len(self.target_paths) # THIS IS BECAUSE OF BATCHNORMS - CHANGE WHEN NOT OVERFITTING!
        return len(self.target_paths)

