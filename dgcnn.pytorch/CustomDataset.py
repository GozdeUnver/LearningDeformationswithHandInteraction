from pathlib import Path
import json
import open3d as o3d
import numpy as np
import torch
import os
import cv2
import open3d
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,split,path=None,deformation_path=None,num_points=2048):
        super().__init__()
        if split=="train":
            f_t=open("./data/train_targets.txt")
            self.target_paths=[line.rstrip() for line in f_t]
            f_i=open("./data/train_inputs.txt")
            self.input_paths=[line.rstrip() for line in f_i]
        
        else:
            self.target_paths=["./data/Sphere/normal/targets/5.obj","./data/Sphere/10_offcenter/targets/5.obj",
            "./data/Sphere/20_offcenter/targets/5.obj",
            "./data/YellowToy01/targets/deformed_2_correspondences_zoomout_2048_total_shape_2048.ply",
            "./data/YellowToy01/targets/deformed_4_correspondences_zoom_2048.ply"]
            self.input_paths=["./data/Sphere/normal/inputs/1.obj",
            "./data/Sphere/10_offcenter/inputs/1.obj",
            "./data/Sphere/20_offcenter/inputs/1.obj",
            "./data/YellowToy01/inputs/non_deformed_2_correspondences_zoom_2048.ply",
            "./data/YellowToy01/inputs/non_deformed_4_correspondences_zoom_2048.ply"]

        #self.num_points=num_points
        self.seg_num_all = 3
        self.seg_start_index = 0
            
    def __getitem__(self, index):
        #index = 0 # THIS IS BECAUSE OF BATCHNORMS - CHANGE WHEN NOT OVERFITTING!
        #input_mesh=np.asarray(open3d.io.read_point_cloud(self.input_paths[index]).points, dtype=np.float32)
        #target_mesh=np.asarray(open3d.io.read_point_cloud(self.target_paths[index]).points, dtype=np.float32)
        #deformed_mesh=np.asarray(open3d.io.read_point_cloud(self.deformation_deformed[index]).points, dtype=np.float32)
        #non_deformed_mesh=np.asarray(open3d.io.read_point_cloud(self.deformation_nondeformed[index]).points, dtype=np.float32)
        if self.input_paths[index][-3:]=="obj":
            input_mesh=np.asarray(o3d.io.read_triangle_mesh(self.input_paths[index]).vertices,dtype=np.float32)
            target_mesh=np.asarray(o3d.io.read_triangle_mesh(self.target_paths[index]).vertices,dtype=np.float32)
        else:
            input_mesh=np.asarray(open3d.io.read_point_cloud(self.input_paths[index]).points, dtype=np.float32)
            target_mesh=np.asarray(open3d.io.read_point_cloud(self.target_paths[index]).points, dtype=np.float32)
       
        deformed_mesh=target_mesh
        non_deformed_mesh=target_mesh
        deformation_vectors = non_deformed_mesh-deformed_mesh
        distances = np.linalg.norm(deformation_vectors, axis=1)
        max_index=np.argmax(distances)
        deformation_vector=deformation_vectors[max_index]
        mesh_index=np.where(input_mesh == non_deformed_mesh[max_index])
        deformations=np.zeros(input_mesh.shape)
        deformations[mesh_index,:] =deformation_vector
        input_mesh = np.concatenate([input_mesh, deformations], axis=1)

        return input_mesh, target_mesh
    
    def __len__(self):
        #return 2 * len(self.target_paths) # THIS IS BECAUSE OF BATCHNORMS - CHANGE WHEN NOT OVERFITTING!
        return len(self.target_paths)

