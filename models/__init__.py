import torch
import torch.nn as nn
from torch import distributions as dist
# from models.render_utils import models
import torch.nn.functional as F
import numpy as np
import os

# def get_renderer(renderer_cfg, **kwargs):
#     ''' 
#     Args:
#         cfg (dict): imported yaml config 
#         device (device): pytorch device
#     '''
#     # setup encoder
#     model_type = renderer_cfg.pop('type')
#     renderer = getattr(models, model_type)(*renderer_cfg.values())
    
#     return renderer

def compose_featmaps(feat_xy, feat_xz, feat_yz):
    H, W = feat_xy.shape[-2:]
    D = feat_xz.shape[-1]

    empty_block = torch.zeros(list(feat_xy.shape[:-2]) + [D, D], dtype=feat_xy.dtype, device=feat_xy.device)
    composed_map = torch.cat(
        [torch.cat([feat_xy, feat_xz], dim=-1),
         torch.cat([feat_yz.transpose(-1, -2), empty_block], dim=-1)], 
        dim=-2
    )
    
    return composed_map, (H, W, D)


def decompose_featmaps(composed_map, sizes):
    H, W, D = sizes
    feat_xy = composed_map[..., :H, :W] # (C, H, W)
    feat_xz = composed_map[..., :H, W:] # (C, H, D)
    feat_yz = composed_map[..., H:, :W].transpose(-1, -2) # (C, W, D)
    return feat_xy, feat_xz, feat_yz