import numpy as np
import open3d as o3d
import pymesh as pm

from scipy.spatial.distance import cdist


Vec2i = o3d.utility.Vector2iVector
Vec3d = o3d.utility.Vector3dVector
Vec3i = o3d.utility.Vector3iVector


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
    # assert len(np.unique(mesh_v_merge_map, axis=0)) == len(cloud_points)

    return mesh_v_merge_map


def replace_vertices_in_mesh(mesh, points, in_place):
    mesh_points = o3d.utility.Vector3dVector(points)
    if in_place:
        mesh.vertices = mesh_points
        return mesh
    else:
        return o3d.geometry.TriangleMesh(mesh_points, mesh.triangles)


def remove_duplicate_vertices(mesh, cloud_points, in_place):
    pm_mesh = pm.form_mesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
    pm_mesh, _ = pm.remove_duplicated_vertices(pm_mesh)

    mesh_to_point_cloud = cdist(pm_mesh.vertices, cloud_points).argmin(axis=1)
    vertices = Vec3d(cloud_points.copy())
    triangles = Vec3i(mesh_to_point_cloud[pm_mesh.faces].copy())

    if in_place:
        mesh.vertices = vertices
        mesh.triangles = triangles
        return mesh
    else:
        return o3d.geometry.TriangleMesh(vertices, triangles)


def pc_to_mesh(mesh, point_cloud, in_place=True, assert_shapes=False):
    mesh_points = np.asarray(mesh.vertices)
    cloud_points = np.asarray(point_cloud.points)

    if assert_shapes:
        assert_intersection_size(mesh_points, cloud_points)

    mesh_v_merge_map = compute_v_merge_map(mesh_points, cloud_points)
    mesh_points = cloud_points[mesh_v_merge_map]
    mesh = replace_vertices_in_mesh(mesh, mesh_points, in_place)
    mesh = remove_duplicate_vertices(mesh, cloud_points, in_place)    
    assert len(mesh.vertices) == len(cloud_points)

    return mesh
