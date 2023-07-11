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
from pymesh import mesh_to_graph, form_mesh as form_pm_mesh


def get_graph_from_mesh(mesh):
    mesh = mesh.triangulate()
    points = np.array(mesh.points)
    faces = np.array(mesh.faces).reshape(-1, 4)[:, 1:]
    pm_mesh = form_pm_mesh(points, faces)

    return mesh_to_graph(pm_mesh)


class CubeDataset(torch.utils.data.Dataset):
    def __init__(self,split,preload=False, path=None,deformation_path=None,num_points=2048):
        super().__init__()
        df_temp=pd.read_csv('data/cube_dataset.csv')
        self.split = split
        if split=="train":
            df=df_temp.loc[df_temp['train_sample'] == True]
        else:
            df=df_temp.loc[df_temp['train_sample'] != True]
        
        df.reset_index(drop=True,inplace=True)
        self.input_paths=df.input.tolist()
        self.target_paths=df["target"].tolist()
        self.contact_points=df["contact_vertex"].astype(int).tolist()
        if preload:
            self.data = []
            for i in range(len(self)):
                if i % 100 == 0:
                    print(f'\rPre-loading dataset: {i}/{len(self)}')
                self.data.append(self._load_item(i))
        else:
            self.data = None

        self.seg_num_all = 3
        self.seg_start_index = 0
            
    def __getitem__(self, index):
        return self.data[index] if self.data is not None else self._load_item(index)
    
    def __len__(self):
        return len(self.target_paths)
    
    def _load_item(self, index):
        input_mesh = pv.read(self.input_paths[index])
        target_mesh = pv.read(self.target_paths[index])
        contact_point=self.contact_points[index]
       
        input_mesh_points = input_mesh.points.copy()
        target_mesh_points = target_mesh.points
        deformations=np.zeros_like(input_mesh_points)
        deformations[contact_point,:] =input_mesh_points[contact_point]-target_mesh_points[contact_point]
        input_graph_embeddings = np.concatenate([input_mesh_points, deformations], axis=1)

        input_graph_vertices, input_graph_edges = get_graph_from_mesh(input_mesh)
        assert np.all(input_graph_vertices == input_mesh_points) 
        input_graph = (
            torch.from_numpy(input_graph_embeddings).to(dtype=torch.float32),
            torch.from_numpy(input_graph_edges).T.to(dtype=torch.int64)
        )

        target_points = torch.from_numpy(target_mesh_points).to(dtype=torch.float32)
        if self.split == 'test':
            input_faces = torch.from_numpy(input_mesh.faces).reshape(-1, 4)[:, 1:]
            return input_faces, input_graph, target_points
        else:
            return input_graph, target_points


