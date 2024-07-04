import torch
import torch.nn as nn

from .unet import UNet3DModel
from .unet_gdf import UNet3DGDFModel

class DiffusionNet(nn.Module):
    def __init__(self, cfg, conditioning_key=None):
        """ init method """
        super().__init__()
        model_type = cfg['model_type']
        self.feed_forward = False
        if model_type == 'UNet3D':
            cfg = cfg['UNet3D_kwargs']
            self.diffusion_net = UNet3DModel(**cfg)
        elif model_type == 'UNet3DGDF':
            self.feed_forward = True
            cfg = cfg['UNet3D_kwargs']
            self.diffusion_net = UNet3DGDFModel(**cfg)
        else:
            raise NotImplementedError(f"{model_type}")
        self.conditioning_key = conditioning_key


    def forward(self, x, t=None, c_concat: list = None, c_crossattn: list = None, sizes: tuple = None):
        # x: should be latent code. shape: (bs X z_dim X d X h X w)
        # sizes: (HxWxD) only for triplane unet
        if self.feed_forward:
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_net(x, context=cc) if sizes == None else self.diffusion_net(x, context=cc, sizes=sizes)
        elif self.conditioning_key is None:
            out = self.diffusion_net(x, t) if sizes == None else self.diffusion_net(x, t, sizes=sizes)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_net(xc, t) if sizes == None else self.diffusion_net(xc, t, sizes=sizes)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_net(x, t, context=cc) if sizes == None else self.diffusion_net(x, t, context=cc, sizes=sizes)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_net(xc, t, context=cc) if sizes == None else self.diffusion_net(xc, t, context=cc, sizes=sizes)
            # import pdb; pdb.set_trace()
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_net(x, t, y=cc) if sizes == None else self.diffusion_net(x, t, y=cc, sizes=sizes)
        else:
            raise NotImplementedError()

        return out