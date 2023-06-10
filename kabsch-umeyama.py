import numpy as np
import open3d as o3d
import meshio

def kabsch_umeyama(A, B,scale=True):
    assert A.shape == B.shape
    n, m = A.shape
    print(n,m)
###########
    EA = np.mean(A, axis=0)
    EB = np.mean(B, axis=0)
    VarA = np.var(A, axis=0).sum()

    H = ((A - EA).T @ (B - EB)) 
    U, D, VT = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])

    R = VT @ S @ U
    c =VarA / np.trace(np.diag(D) @ S)
    t = EA - R @ EB.T


    return R, c, t



#not deformed
A_correspondence=np.load("./data/mesh1_correspondences.npy")
#deformed
B_correspondence=np.load("./data/mesh2_correspondences.npy")
R, c, t = kabsch_umeyama(A_correspondence, B_correspondence)
print(R,c,t)

#B_pcd_updated = np.array([t+ R @ b for b in B_correspondence])
B_pcd_updated=np.matmul(B_correspondence,R.T)+t

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(B_pcd_updated)
o3d.io.write_point_cloud("temp.ply", pcd)


