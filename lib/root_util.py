import torch
import trimesh
import numpy as np
import point_cloud_utils as pcu

def save_obj(mesh_path, verts):
    file = open(mesh_path, 'w')    
    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    file.close()


def projection(points, calibrations, is_head=True):
    '''
    Compute the orthogonal projections of 3D points into the image plane by given projection matrix
    :param points: [B, 3, N] Tensor of 3D points
    :param calibrations: [B, 4, 4] Tensor of projection matrix
    :param transforms: [B, 2, 3] Tensor of image transform matrix
    :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image plane
    '''
    if calibrations.ndim == 2:
        calibrations = calibrations[None, ...]

    print("calib: ", calibrations.shape)
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]

    pts = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    xy = pts[:, :2, :]
    if is_head:
        xy = pts[:, :2, :] / pts[:, 2:3, :]
     
    
    xyz = torch.cat([xy, pts[:, 2:3, :]], 1)

    return xyz

def get_3d_roots(pts_2d, head_mesh_path, calib_path, sample_num=200000, is_head=True):
    head_mesh = trimesh.load(head_mesh_path)
    sampling_points, _ = trimesh.sample.sample_surface_even(head_mesh, sample_num)
    if calib_path.endswith(".pth"):
        calib = torch.load(calib_path).float().cuda()
    else: 
        calib = torch.from_numpy(np.load(calib_path)).float().cuda()

    pts_z = np.squeeze(sampling_points[:, 2:3])
    pts_y = np.squeeze(sampling_points[:, 1:2])

    if is_head:
        # index_selected = (np.array(pts_z) > 0.1) & ((np.array(pts_y) < 0.8) & (np.array(pts_y) > 0.3))
        index_selected = (pts_z > 0) & ((pts_y < 0.7) & (pts_y > 0.2)) 
        # index_selected = (pts_z > -0.2) & ((pts_y < 0.7) & (pts_y > 0.2))   # just for beeler
        sampling_points = sampling_points[index_selected]
    else:
        sampling_points = sampling_points

    points_3d = torch.Tensor(sampling_points.T).float().unsqueeze(0).cuda()

    xyz = projection(points_3d, calib, is_head)

    xy = xyz[0, :2, :]
    x_coor = xy.squeeze().cpu().detach().numpy().T[:,0]
    y_coor = xy.squeeze().cpu().detach().numpy().T[:,1]

    pts_3d_proj = np.concatenate([y_coor[:,None], x_coor[:,None]], axis=1).astype(np.int32)
    pts_2d_cat = np.concatenate([pts_2d, np.zeros([pts_2d.shape[0], 1])], axis=1).astype(np.float64)
    pts_3d_proj_cat = np.concatenate([pts_3d_proj, np.zeros([pts_3d_proj.shape[0], 1])], axis=1).astype(np.float64)

    K = 1
    dists_a_to_b, corrs_a_to_b = pcu.k_nearest_neighbors(pts_2d_cat, pts_3d_proj_cat, K)
    pts_3d_index_selected_by_z = corrs_a_to_b
    return sampling_points[pts_3d_index_selected_by_z]