
import open3d as o3d
import numpy as np

def NumpyToPCD(xyz):
    """ convert numpy ndarray to open3D point cloud 
    Args:
        xyz (ndarray): 
    Returns:
        [open3d.geometry.PointCloud]: 
    """

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    return pcd


def PCDToNumpy(pcd):
    """  convert open3D point cloud to numpy ndarray
    Args:
        pcd (open3d.geometry.PointCloud): 
    Returns:
        [ndarray]: 
    """

    return np.asarray(pcd.points)

def get_room_walls(xyz, wall_ind, distance=0.1, init_n=3, iter=200, max_num=4):

    wall_ind = wall_ind.astype('bool')

    walls = []

    remain_wall_ind = np.where(wall_ind)[0]
    remain_wall_xyz = xyz[wall_ind]

    for _ in range(max_num):
        
        if remain_wall_xyz.shape[0] < 10000:
            break

        wall_pcd = NumpyToPCD(remain_wall_xyz)

        w, index = wall_pcd.segment_plane(distance, init_n, iter)

        index = np.asarray(index)
        # index = np.array(index)
        cur_wall_ind = remain_wall_ind[index]

        cur_wall_mask = np.zeros(len(xyz)).astype(bool)
        cur_wall_mask[cur_wall_ind] = True

        walls.append(cur_wall_mask)


        remain_mask = np.ones(len(remain_wall_xyz)).astype(bool)
        remain_mask[index] = False

        remain_wall_xyz = remain_wall_xyz[remain_mask]
        remain_wall_ind = remain_wall_ind[remain_mask]

    return walls