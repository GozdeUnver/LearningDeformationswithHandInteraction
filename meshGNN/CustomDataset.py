from pathlib import Path

import numpy as np
import torch
import open3d as o3d
import os

from pymesh import mesh_to_graph, form_mesh as form_pm_mesh

from utils.reduce_mesh_vertices import pc_to_mesh


def combine_mesh_vectors(input_mesh, non_deformed_mesh, deformed_mesh):
    input_mesh_points = np.asarray(input_mesh.vertices).copy()
    deformation_vectors = non_deformed_mesh-deformed_mesh
    distances = np.linalg.norm(deformation_vectors, axis=1)
    max_index = np.argmax(distances)
    deformation_vector=deformation_vectors[max_index]
    mesh_index, _ = np.where(input_mesh_points == non_deformed_mesh[max_index])
    deformations=np.zeros(input_mesh_points.shape)
    deformations[mesh_index,:] = deformation_vector

    return np.concatenate([input_mesh_points, deformations], axis=1)

def get_graph_from_mesh(mesh):
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.triangles)
    pm_mesh = form_pm_mesh(vertices, faces)

    return mesh_to_graph(pm_mesh)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, num_points=2048, preload=False, flavor='all'):
        super().__init__()

        self.deformation_nondeformed_paths=["../data/pointcloud_sampled/YellowToy01/deformations/non_deformed_2_correspondences_zoom_2048_paired_648.ply"]
        self.deformation_deformed_paths=["../data/pointcloud_sampled/YellowToy01/deformations/deformed_2_correspondences_zoom_2048_paired_648.ply"]
        self.target_mesh_paths = ['../data/pointcloud_sampled/YellowToy01/targets/yellow_push_toy_2_70000.obj']
        self.target_pc_paths = ['../data/pointcloud_sampled/YellowToy01/targets/deformed_2_correspondences_zoom_2048.ply']
        self.input_mesh_paths = ['../data/pointcloud_sampled/YellowToy01/inputs/yellow_push_toy_1_70000.obj']
        self.input_pc_paths = ['../data/pointcloud_sampled/YellowToy01/inputs/non_deformed_2_correspondences_zoom_2048.ply']

        assert flavor in ['train', 'test']
        self.flavor = flavor

        if preload:
            self.data = [self._load_graphs(index) for index in range(len(self.input_mesh_paths))]
        else:
            self.data = None

        self.num_points=num_points
        self.seg_num_all = 3
        self.seg_start_index = 0
            
    def __getitem__(self, index):
        index = 0 # THIS IS BECAUSE OF BATCHNORMS - CHANGE WHEN NOT OVERFITTING!
        if self.data is None:
            return self._load_graphs(index)
        else:
            return self.data[index]

    def __len__(self):
        return 2 * len(self.input_mesh_paths) # THIS IS BECAUSE OF BATCHNORMS - CHANGE WHEN NOT OVERFITTING!
        return len(self.input_mesh_paths)
    
    def _load_graphs(self, index):
        deformed_mesh=np.asarray(o3d.io.read_point_cloud(self.deformation_deformed_paths[index]).points, dtype=np.float32)
        non_deformed_mesh=np.asarray(o3d.io.read_point_cloud(self.deformation_nondeformed_paths[index]).points, dtype=np.float32)

        input_mesh = o3d.io.read_triangle_mesh(self.input_mesh_paths[index])
        input_pc = o3d.io.read_point_cloud(self.input_pc_paths[index])
        pc_to_mesh(input_mesh, input_pc)

        target_mesh = o3d.io.read_triangle_mesh(self.target_mesh_paths[index])
        target_pc = o3d.io.read_point_cloud(self.target_pc_paths[index])
        pc_to_mesh(target_mesh, target_pc)

        input_mesh_points = np.asarray(input_mesh.vertices)
        input_graph_embeddings = combine_mesh_vectors(input_mesh, non_deformed_mesh, deformed_mesh)
        assert np.all(input_graph_embeddings[:, :3] == input_mesh_points)

        input_graph_vertices, input_graph_edges = get_graph_from_mesh(input_mesh)
        assert np.all(input_graph_vertices == input_mesh_points)
        input_graph = (
            torch.from_numpy(input_graph_embeddings).to(dtype=torch.float32),
            torch.from_numpy(input_graph_edges).T.to(dtype=torch.int64)
        )
        
        target_points = torch.from_numpy(np.asarray(target_mesh.vertices)).to(dtype=torch.float32)
        if self.flavor == 'test':
            input_triangles = torch.from_numpy(np.asarray(input_mesh.triangles))
            return input_triangles, input_graph, target_points
        else:
            return input_graph, target_points
    