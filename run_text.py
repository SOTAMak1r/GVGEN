import torch
from PIL import Image
from models.igen import Condition_DiffusionModel
from opt import get_opts, get_cfgs
from external.clip import tokenize
import numpy as np
import random

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import time
from plyfile import PlyData, PlyElement
from gaussian_renderer import de_normalize_brief


if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False, precision=5)

    hparams = get_opts()
    print(str(hparams))

    random.seed(hparams.seed)
    np.random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    torch.cuda.manual_seed_all(hparams.seed)  # 对多个GPU进行设置（如果有的话）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dm_cfg    = get_cfgs(hparams.dm_cfg_path)
    recon_cfg = get_cfgs(hparams.recon_cfg_path)

    with torch.no_grad():
        dm_pipe = Condition_DiffusionModel.load_from_checkpoint(
            checkpoint_path=hparams.dm_ckpt_path,
            strict=False,
            opts=hparams,
            cfg=dm_cfg,
            mode='test'
        ).cuda()
        dm_pipe = dm_pipe.eval()

        recon_pipe = Condition_DiffusionModel.load_from_checkpoint(
            checkpoint_path=hparams.recon_ckpt_path,
            strict=False,
            opts=hparams,
            cfg=recon_cfg,
            mode='test'
        ).cuda()
        recon_pipe = recon_pipe.eval()

        print(f'[INFO] FINISH LOAD TWO MODEL !')

        print(f'[text input] {hparams.text_input}')

        text = hparams.text_input
        start_time = time.time()
        token = tokenize(text)

        
        y, pred_y0s = dm_pipe.inference(
                condition=token,
                ddim_steps=100, 
            )
        GDF = y[0].permute(1, 2, 3, 0)

        y, pred_y0s = recon_pipe.inference(
                condition=token,
                gdf_cond=GDF, 
            )

        end_time = time.time()
        print(f'\n[INFO] time cost = {end_time - start_time} seconds')


        pred_y0s.append(y)
                        
        recover_x = recon_pipe.display(
                pred_y0s, 
                render_path='.', 
            )

        # save splat
        ply_path = './sample.ply'

        _xyz, _features_dc, _scaling, _rotation, _opacity, _ = de_normalize_brief(recover_x)

        OPACITY_THRESHOLD = -5
        opacity_unsqueeze = _opacity.squeeze(-1)
        _xyz = _xyz[opacity_unsqueeze > OPACITY_THRESHOLD]
        _features_dc = _features_dc[opacity_unsqueeze > OPACITY_THRESHOLD]
        _scaling = _scaling[opacity_unsqueeze > OPACITY_THRESHOLD]
        _rotation = _rotation[opacity_unsqueeze > OPACITY_THRESHOLD]
        _opacity = _opacity[opacity_unsqueeze > OPACITY_THRESHOLD]

        max_sh_degree = 1
        extra = 3*(max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((_xyz.shape[0], extra))
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
        _features_rest = torch.tensor(features_extra).transpose(1, 2).contiguous()

        def construct_list_of_attributes():
            l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
            # All channels except the 3 DC
            for i in range(_features_dc.shape[1]*_features_dc.shape[2]):
                l.append('f_dc_{}'.format(i))
            for i in range(_features_rest.shape[1]*_features_rest.shape[2]):
                l.append('f_rest_{}'.format(i))
            l.append('opacity')
            for i in range(_scaling.shape[1]):
                l.append('scale_{}'.format(i))
            for i in range(_rotation.shape[1]):
                l.append('rot_{}'.format(i))
            return l

        xyz = _xyz.detach().cpu().numpy() # * 100
        normals = np.zeros_like(xyz)
        f_dc = _features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = _features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = _opacity.detach().cpu().numpy()
        scale = _scaling.detach().cpu().numpy() # + 9.21
        rotation = _rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(ply_path)