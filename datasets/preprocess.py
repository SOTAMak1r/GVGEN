import torch

import numpy as np

import torch.nn.functional as F
import pickle

def fourd_to_threed(rot):
    r = 1.0 
    
    w = rot[..., 0:1]
    x = rot[..., 1:2]
    y = rot[..., 2:3]
    z = rot[..., 3:4]
    
    theta = torch.asin(torch.clamp(torch.sqrt(x**2 + y**2 + z**2) / r, min=-1, max=1))
    phi   = torch.atan2(y, x)
    psi   = torch.atan2(torch.sqrt(x**2 + y**2), z)

    return torch.cat((theta, phi, psi), dim=-1)

def threed_to_fourd(three):
    r = 1.0

    theta = three[..., 0:1]
    phi   = three[..., 1:2]
    psi   = three[..., 2:3]

    w = r * torch.cos(theta)
    z = r * torch.sin(theta) * torch.cos(psi)
    x = r * torch.sin(theta) * torch.sin(psi) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(psi) * torch.sin(phi)

    return torch.cat((w, x, y, z), dim=-1).to(three.device)

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3))

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float)
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L, R

def quaternion_to_matrix(quaternions):
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:

    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(*batch_dim, 9), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(q_abs.new_tensor(0.1)))

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
    ].reshape(*batch_dim, 4)


def get_axis_trans_matrix(sorted_indices):
    ret = torch.ones((sorted_indices.shape[0], 3, 3))
    idx = torch.tensor((0, 1, 2))#.unsqueeze(0)
    dame = torch.sum(sorted_indices != idx, dim=-1)#.unsqueeze(-1)

    ret[dame==2, :, 1] = -1
    
    return ret

def cov_trans(scale, rotation):
    L, R = build_scaling_rotation(scale, rotation)

    sorted_indices = torch.argsort(scale, dim=1, descending=True)
    
    sorted_R = R.permute(0, 2, 1) 
    
    sorted_R = torch.gather(sorted_R, dim=1, index=sorted_indices.unsqueeze(-1).expand(-1, -1, sorted_R.size(-1)))
    sorted_R = sorted_R.permute(0, 2, 1)

    sorted_S = torch.gather(scale, dim=1, index=sorted_indices)

    sorted_R *= get_axis_trans_matrix(sorted_indices)

    sorted_R = matrix_to_quaternion(sorted_R)

    # double_check_R = quaternion_to_matrix(sorted_R)

    return sorted_S, sorted_R