import torch
import numpy as np

def inv_transform_bfm_mesh(bfm_mesh, transform):
    scale, offset = transform
    offset = offset[0]    
    offset_new = - offset
    scale_new  = 1 / scale
    out_mesh = bfm_mesh.scale_verts(scale_new)
    out_mesh = out_mesh.offset_verts(offset_new)

    return out_mesh

def get_3d_lmk_bbox(lmk_3d):
    lmk3d_min_xyz = np.min(lmk_3d, axis=0, keepdims=True)
    lmk3d_max_xyz = np.max(lmk_3d, axis=0, keepdims=True)
    center = (lmk3d_min_xyz + lmk3d_max_xyz) * 0.5
    bbox_length = np.max(lmk3d_max_xyz - lmk3d_min_xyz)

    return center, bbox_length


def normalize_lmk(src_lmk3d_pts, ref_lmk3d_pts):
    ref_c, ref_l = get_3d_lmk_bbox(ref_lmk3d_pts)
    _, src_l = get_3d_lmk_bbox(src_lmk3d_pts)

    scale = ref_l / src_l
    src_lmk3d_pts_normalize = src_lmk3d_pts * scale
    src_c, _ = get_3d_lmk_bbox(src_lmk3d_pts_normalize)
    
    trans = ref_c - src_c
    src_lmk3d_pts_normalize += trans
    
    ### 51 landmarks
    # src_lmk3d_pts_normalize = src_lmk3d_pts_normalize[17:, :]
    # ref_lmk3d_pts = ref_lmk3d_pts[17:, :]

    scale = torch.Tensor([scale])
    trans = torch.from_numpy(trans[0])

    return src_lmk3d_pts_normalize, ref_lmk3d_pts, (scale, trans)


def transform_mesh(mesh, transform):
    scale, offset = transform
    out_mesh = mesh.scale_verts(scale)
    out_mesh = out_mesh.offset_verts(offset)
    return out_mesh


def inv_transform_verts(verts, transform):
    scale, offset = transform
    scale, offset = scale.numpy(), offset.numpy()
    offset_new = -1.0 * offset
    scale_new  = 1.0 / scale
    
    verts += offset_new
    verts *= scale_new
    
    return verts

def inv_total_transform(verts, transform_fn):
    transform = torch.load(transform_fn)
    norm_lmk_transf = transform["norm_lmk"]
    align_lmk_transf = transform["align_lmk"]
    verts = inv_transform_verts(verts, align_lmk_transf)
    verts = inv_transform_verts(verts, norm_lmk_transf)
    
    return verts


def transform_lmk(lmk3d, transform):
    if type(lmk3d) == np.ndarray:
        lmk3d_pts = torch.from_numpy(lmk3d)
    else:
        assert type(lmk3d) == torch.Tensor, "Type Error!"
        lmk3d_pts = lmk3d
        
    lmk3d_pts *= transform[0]
    lmk3d_pts += transform[1]
    return lmk3d_pts

