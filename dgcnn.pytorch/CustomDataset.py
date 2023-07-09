from pathlib import Path
import json
import open3d as o3d
import numpy as np
import torch
import os
import cv2
import open3d
import pandas as pd
import pyvista as pv

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,split,path=None,deformation_path=None,num_points=2048):
        super().__init__()
        df_temp=pd.read_csv("./data/cube_dataset.csv")
        if split=="train":
            df=df_temp.loc[df_temp['train_sample'] == True]
        else:
            df=df_temp.loc[df_temp['train_sample'] == False]

        self.input_paths=df["input"]
        self.target_paths=df["target"]
        self.contact_points=df["contact_vertex"].astype(int)

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
            input_mesh=np.asarray(pv.read(self.input_paths[index]).points,dtype=np.float16)
            target_mesh=np.asarray(pv.read(self.target_paths[index]).points, dtype=np.float16)
            contact_point=self.contact_points[index]
       
        
        deformations=np.zeros(input_mesh.shape)
        deformations[contact_point,:] =input_mesh[contact_point]-target_mesh[contact_point]
        input_mesh = np.concatenate([input_mesh, deformations], axis=1)

        return input_mesh, target_mesh
    
    def __len__(self):
        #return 2 * len(self.target_paths) # THIS IS BECAUSE OF BATCHNORMS - CHANGE WHEN NOT OVERFITTING!
        return len(self.target_paths)

