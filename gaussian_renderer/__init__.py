#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from einops import rearrange
import torch
import math
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.sh_utils import eval_sh
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

def get_grid(voxel_size=32, world_size=0.5):
    N = voxel_size
    interval = N - 1
    grid_unseen_xyz = torch.zeros((N, N, N, 3))
    for i in range(N):
        grid_unseen_xyz[i, :, :, 0] = i
    for j in range(N):
        grid_unseen_xyz[:, j, :, 1] = j
    for k in range(N):
        grid_unseen_xyz[:, :, k, 2] = k
    grid_unseen_xyz -= (interval / 2.0)
    grid_unseen_xyz /= (interval / 2.0) / world_size
    grid_unseen_xyz = grid_unseen_xyz.reshape((-1, 3))
    return grid_unseen_xyz

def denormalize_attr(feats, data_dist_scale=1, mean_var=False):
    
    maxs = ( 0.15,  5,  -2,  1,  7)
    mins = (-0.15, -2,  -6, -1, -7)

    if mean_var:
        feats = [
            torch.clamp(feats[i], min=mins[i], max=maxs[i])
            for i in range(len(feats))
        ]
    else:
        feats = [
            (feats[i] + data_dist_scale) * (maxs[i] - mins[i]) / (2 * data_dist_scale) + mins[i] 
            for i in range(len(feats))
        ]

    return feats

def de_normalize(pc, data_dist_scale=1, mean_var=False):
    scaling_activation = torch.exp
    opacity_activation = torch.sigmoid
    rotation_activation = torch.nn.functional.normalize
    voxel_xyz = get_grid().to(pc.device)
    
    channels = [0, 3, 6, 9, 13, 14]
    attributes = [pc[..., channels[i]:channels[i+1]] for i in range(5)]

    xyz, features, scales, rotations, opacity = denormalize_attr(attributes, data_dist_scale=data_dist_scale, mean_var=mean_var)
    
    active_sh_degree = 0
    xyz += voxel_xyz
    features = features.reshape((-1, 1, 3))
    scales = scaling_activation(scales)
    rotations = rotation_activation(rotations)
    opacity = opacity_activation(opacity)

    
    return xyz, features, scales, rotations, opacity, active_sh_degree

def de_normalize_brief(pc, data_dist_scale=1, mean_var=True):
    # scaling_activation = torch.exp
    # opacity_activation = torch.sigmoid
    # rotation_activation = torch.nn.functional.normalize
    voxel_xyz = get_grid().to(pc.device)
    channels = [0, 3, 6, 9, 13, 14]
    attributes = [pc[..., channels[i]:channels[i+1]] for i in range(5)]

    xyz, features, scales, rotations, opacity = denormalize_attr(attributes, data_dist_scale=data_dist_scale, mean_var=mean_var)
    active_sh_degree = 0
    xyz += voxel_xyz
    features = features.reshape((-1, 1, 3))
    # scales = scaling_activation(scales)
    # rotations = rotation_activation(rotations)
    # opacity = opacity_activation(opacity)

    return xyz, features, scales, rotations, opacity, active_sh_degree

def render(viewpoint_camera, pc, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, data_dist_scale=1., mean_var=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    xyz, features, scales, rotations, opacity, active_sh_degree = de_normalize(pc, data_dist_scale=data_dist_scale, mean_var=mean_var)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    R, T = viewpoint_camera[:3, :3], viewpoint_camera[-1, :3]
    
    FoVx = FoVy = viewpoint_camera[-1, -1]
    image_height = image_width = 512
    world_view_transform = getWorld2View2(R, T).transpose(0, 1)
    projection_matrix = getProjectionMatrix(znear=0.01, zfar=100, fovX=FoVx, fovY=FoVy).transpose(0,1).to(viewpoint_camera.device)
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]
    # Set up rasterization configuration
    tanfovx = math.tan(FoVx * 0.5) # / 2 * 1.414
    tanfovy = math.tan(FoVy * 0.5) # / 2 * 1.414
    if bg_color[0] == 142 / 255.0:
        tanfovx = tanfovx / 2 * 1.414
        tanfovy = tanfovy / 2 * 1.414
    raster_settings = GaussianRasterizationSettings(
        image_height=image_height,
        image_width=image_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=active_sh_degree,
        campos=camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = xyz
    means2D = screenspace_points
    opacity = opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = None
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        shs = features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
