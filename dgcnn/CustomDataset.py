from pathlib import Path
import json

import numpy as np
import torch
import os
import cv2
import open3d
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,path=None,deformation_path=None,num_points=2048):
        super().__init__()
        self.input_paths=["./pointcloud_sampled/BlueToy02/inputs/blue_push_toy_1_70000_sampled.ply"]
        self.target_paths=["./pointcloud_sampled/BlueToy02/targets/blue_push_toy_2_70000_sampled.ply"]
        self.deformation_vertex=["./pointcloud_sampled/BlueToy02/deformations/blue_deformation_1_2.txt"]
        self.num_points=num_points
        self.seg_num_all = 3
        self.seg_start_index = 0
        self.label=np.ones((len(self.input_paths),1)) #TODO : WHEN DIFFERENT CLASSES ARE TRAINED THIS SHOULD BE CHANGED
            
    def __getitem__(self, index):
        index = 0 # THIS IS BECAUSE OF BATCHNORMS - CHANGE WHEN NOT OVERFITTING!
        input_mesh=np.asarray(open3d.io.read_point_cloud(self.input_paths[index]).points, dtype=np.float32)
        target_mesh=np.asarray(open3d.io.read_point_cloud(self.target_paths[index]).points, dtype=np.float32)
        f=open("./pointcloud_sampled/BlueToy02/deformations/blue_deformation_1_2.txt","r").readlines()
        for i,line in enumerate(f):
            if i==2:
                indexes=line.split()
                nondeformed_idx=float(indexes[0])
                deformed_idx=float(indexes[1])
            elif i==3:
                deformation_distance=float(line)
        return {"input":input_mesh,"target":target_mesh,"label":self.label[index],"nondeformed_idx":nondeformed_idx,
                "deformed_idx":deformed_idx,"deformation_distance":deformation_distance}
    
    def __len__(self):
        return 2 * len(self.target_paths) # THIS IS BECAUSE OF BATCHNORMS - CHANGE WHEN NOT OVERFITTING!
        return len(self.target_paths)

