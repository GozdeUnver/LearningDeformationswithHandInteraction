from pathlib import Path

import numpy as np
import torch
import open3d as o3d
import pymesh as pm

from scipy.spatial.distance import cdist


def assert_intersection_size(arr1, arr2):
    # https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
    _, ncols = arr1.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
        'formats':ncols * [arr1.dtype]}
    common_points = np.intersect1d(arr1.view(dtype), arr2.view(dtype))
    common_points = common_points.view(arr1.dtype).reshape(-1, ncols)
    assert common_points.shape == arr1.shape


def compute_v_merge_map(mesh_points, cloud_points):
    mesh_v_merge_map = np.zeros(len(mesh_points), dtype=int)
    distances = cdist(mesh_points, cloud_points)
    mesh_v_merge_map = distances.argmin(axis=1)
    assert len(np.unique(mesh_v_merge_map, axis=0)) == len(cloud_points)


def replace_vertices_in_mesh(mesh, points, in_place):
    mesh_points = o3d.utility.Vector3dVector(points)
    if in_place:
        mesh.vertices = mesh_points
        return mesh
    else:
        return o3d.geometry.TriangleMesh(mesh_points, mesh.triangles)


def remove_duplicate_vertices(mesh, in_place):
    pm_mesh = pm.form_mesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
    pm_mesh = pm.remove_duplicated_vertices(pm_mesh)

    vertices = o3d.utility.Vector3dVector(pm_mesh.vertices)
    faces = o3d.utility.Vector3iVector(pm_mesh.faces)
    if in_place:
        mesh.vertices = vertices
        mesh.triangles = faces
        return mesh
    else:
        return o3d.geometry.TriangleMesh(vertices, faces)


def pc_to_mesh(mesh, point_cloud, in_place=True, assert_shapes=False):
    mesh_points = np.asarray(mesh.vertices)
    cloud_points = np.asarray(point_cloud.points)

    if assert_shapes:
        assert_intersection_size(mesh_points, cloud_points)

    mesh_v_merge_map = compute_v_merge_map(mesh_points, cloud_points)
    mesh_points = cloud_points[mesh_v_merge_map]
    mesh = replace_vertices_in_mesh(mesh, mesh_points, in_place)
    mesh = remove_duplicate_vertices(mesh, in_place)    
    assert len(mesh.vertices) == len(cloud_points)

    return mesh


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,path=None,deformation_path=None,num_points=2048):
        super().__init__()
        self.target_paths = ["./data/pointcloud_sampled/YellowToy01/targets/deformed_2_correspondences_zoom_2048.ply"]
        self.input_paths = ["./data/pointcloud_sampled/YellowToy01/inputs/non_deformed_2_correspondences_zoom_2048.ply"]
        self.deformation_nondeformed=["./data/pointcloud_sampled/YellowToy01/deformations/non_deformed_2_correspondences_zoom_2048_paired_648.ply"]
        self.deformation_deformed=["./data/pointcloud_sampled/YellowToy01/deformations/deformed_2_correspondences_zoom_2048_paired_648.ply"]
        self.num_points=num_points
        self.seg_num_all = 3
        self.seg_start_index = 0
            
    def __getitem__(self, index):
        index = 0 # THIS IS BECAUSE OF BATCHNORMS - CHANGE WHEN NOT OVERFITTING!
        input_mesh=np.asarray(o3d.io.read_point_cloud(self.input_paths[index]).points, dtype=np.float32)
        target_mesh=np.asarray(o3d.io.read_point_cloud(self.target_paths[index]).points, dtype=np.float32)
        deformed_mesh=np.asarray(o3d.io.read_point_cloud(self.deformation_deformed[index]).points, dtype=np.float32)
        non_deformed_mesh=np.asarray(o3d.io.read_point_cloud(self.deformation_nondeformed[index]).points, dtype=np.float32)
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
        return 2 * len(self.target_paths) # THIS IS BECAUSE OF BATCHNORMS - CHANGE WHEN NOT OVERFITTING!
        return len(self.target_paths)

