import numpy as np
import open3d as o3d
import meshio

def kabsch_umeyama(A, B):
    assert A.shape == B.shape
    n, m = A.shape

    EA = np.mean(A, axis=0)
    EB = np.mean(B, axis=0)
    VarA = np.mean(np.linalg.norm(A - EA, axis=1) ** 2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * m)
    
    if d<0:
        S[-1,-1]=d

    R = U @ S @ VT
    #R = U @ VT
    c = VarA / np.trace(np.diag(D) @ S)
    t = EA - c * R @ EB
    return R, c, t



#not deformed
A_correspondence=np.load("./data/mesh1_correspondences.npy")
#deformed
B_correspondence=np.load("./data/mesh2_correspondences.npy")
R, c, t = kabsch_umeyama(A_correspondence, B_correspondence)
print(R,c,t)
B = np.asarray(o3d.io.read_triangle_mesh("./leo_meshes/toy_deformed/toy_deformed.obj",enable_post_processing=True).vertices)

#B=meshio.read("./leo_meshes/toy_deformed/toy_deformed.obj")


B_pcd_updated = np.array([t + c *R @ b for b in B_correspondence])
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(B_pcd_updated)
o3d.io.write_point_cloud("./leo_meshes/temp.ply", pcd)


