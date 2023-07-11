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


class CubeDataset(torch.utils.data.Dataset):
    def __init__(self, split, split_vectors=False, preload=False):
        super().__init__()
        df_temp=pd.read_csv("data/cube_dataset.csv")
        if split=="train":
            df=df_temp.loc[df_temp['train_sample'] == True]
        else:
            df=df_temp.loc[df_temp['train_sample'] != True]
        df.reset_index(drop=True,inplace=True)
        self.input_paths=df.input.tolist()
        self.target_paths=df["target"].tolist()
        self.contact_points=df["contact_vertex"].astype(int).tolist()

        self.seg_num_all = 3
        self.seg_start_index = 0

        self.split_vectors = split_vectors

        if preload:
            self.data = []
            for i in range(len(self)):
                self.data.append(self._load_item(i))
                if i % 100 == 0:
                    print(f'\rPre-loading: {i}/{len(self)}')
        else:
            self.data = None
        
            
    def __getitem__(self, index):
        return self._load_item(index) if self.data is None else self.data[index]
    
    def _load_item(self, index):
        input_mesh=np.asarray(pv.read(self.input_paths[index]).points,dtype=np.float16)
        target_mesh=np.asarray(pv.read(self.target_paths[index]).points, dtype=np.float16)
        contact_point=self.contact_points[index]
       
        deformations=np.zeros(input_mesh.shape)
        deformations[contact_point,:] =input_mesh[contact_point]-target_mesh[contact_point]

        if self.split_vectors:
            return (input_mesh, deformations), target_mesh
        else:
            input_mesh = np.concatenate([input_mesh, deformations], axis=1)
            return inpute_mesh, target_mesh

        return input_mesh, target_mesh

    
    def __len__(self):
        return len(self.target_paths)
