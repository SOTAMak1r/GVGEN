import torch
import os
import imageio.v2 as imageio
import numpy as np
import PIL
from PIL import Image
from einops import rearrange
from gaussian_renderer import render
import math
import torch.nn.functional as F

from pytorch_lightning import LightningModule
from functools import partial

from models.diffusion_networks.network import DiffusionNet
from models.clip_networks.network import CLIPImageEncoder, CLIPTextEncoder

from models.diffusion_networks.samplers.ddim import DDIMSampler

from datasets.preprocess import *
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from utils import *

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def extract(v, t, x_shape):
    out = torch.gather(v, index=t, dim=0).float()
    return out.contiguous().view([t.shape[0]] + [1] * (len(x_shape) -1))


def get_test_poses():

    def get_camera_positions_on_sphere(radius, center_point, num_cameras):
        cameras = []
        for i in range(num_cameras):

            x = -0.5
            radius_at_height = np.sqrt(1 - x * x) 

            theta = np.pi * 2 * i /  num_cameras

            z = np.cos(theta) * radius_at_height
            y = np.sin(theta) * radius_at_height
            camera_position = np.array([y, z, x]) * radius + np.array(center_point)
            cameras.append(camera_position)

        return cameras

    def get_rotation_matrix_to_look_at(target_point, camera_position):
        forward = target_point - camera_position
        forward = forward / np.linalg.norm(forward)

        right = np.array([0, 0, 1]) 
        right = right - np.dot(right, forward) * forward
        right = right / np.linalg.norm(right)

        up = np.cross(forward, right)

        rotation_matrix = np.column_stack((right, up, -forward))

        return rotation_matrix

    R = 0.02  
    center_point = (0, 0, 2) 
    image_num = 60 

    camera_positions = get_camera_positions_on_sphere(R, center_point, image_num)

    target_point = np.array(center_point)

    cam_infos = []
    
    fovx = 49.1 * 3.14159 / 180
    fovx = 40 * 3.14159 / 180
    for idx in range(image_num):
    
        rotation_matrix = get_rotation_matrix_to_look_at(target_point, camera_positions[idx])
        R = rotation_matrix
        T = camera_positions[idx]
        fovy = focal2fov(fov2focal(fovx, 512), 512)
        FovY = fovy 
        FovX = fovx
        cam = np.zeros((1, 4, 4))
        cam[0, :3, :3] = R
        cam[0, -1, :3] = T
        cam[0, -1, -1] = FovX
        cam_infos.append(cam)
    return torch.tensor(np.concatenate(cam_infos, axis=0))



def get_data_type(in_channels, out_channels):
    if in_channels == 1 and out_channels == 1:
        return 'gdf'
    
    return 'gs'


class Condition_DiffusionModel(LightningModule):
    def __init__(
        self, 
        opts,
        cfg,
        mode='train',
        # volume_type: grid triplane
    ):
        super(Condition_DiffusionModel, self).__init__()

        self.clamp_min, self.clamp_max = -5, 5
        self.data_dist_scale = 1.
        
        self.mode = mode
        self.opts = opts

        self.cond_model_config = cfg['cond_model_config']
        self.unet_config       = cfg['unet_config']
        self.df_model_config   = cfg['df_model_config']
        self.render_config     = cfg['render_config']

        self.volume_type       = cfg['volume_type']
        
        self.ddim_steps = 100
        self.df_shape = self.df_model_config['df_shape']

        cond_model_type = self.cond_model_config.get('model_type', 'clip')
        cond_mode       = self.cond_model_config.get('cond_mode', 'image')
        print(f'[INFO] condition model is {cond_model_type, cond_mode}')

        self.cond_model_type = cond_model_type
        self.cond_mode = cond_mode

        if cond_mode == 'image':
            self.cond_model = CLIPImageEncoder(model=self.cond_model_config['model'])
            self.cond_model.to(self.device)
        else:
            self.cond_model = CLIPTextEncoder(model=self.cond_model_config['model'])
            self.cond_model.to(self.device)
                
        self.data_type = get_data_type(
                self.unet_config['UNet3D_kwargs']['in_channels'], 
                self.unet_config['UNet3D_kwargs']['out_channels']
            )
        print(f'[INFO] Diffusion data type is {self.data_type}')

        # init diffusion backbone
        self.df_model = DiffusionNet(self.unet_config, conditioning_key=self.df_model_config['conditioning_key'])
        self.df_model.to(self.device)

        
        if self.unet_config['model_type'] != "UNet3DGDF":
            self.num_timesteps = self.df_model_config['T']
            
            self.parameterization = self.df_model_config['parameterization']

            self.piecewise_beta = self.df_model_config.get('piecewise_beta', False)
            if self.piecewise_beta:
                betas = []
                st_beta = self.df_model_config['beta_1']
                for i in range(len(self.df_model_config['beta_list'])):
                    beta_slice = torch.linspace(st_beta, self.df_model_config['beta_list'][i],  self.df_model_config['T_list'][i])
                    st_beta = self.df_model_config['beta_list'][i]
                    betas.append(beta_slice)
                self.register_buffer('betas', torch.cat(betas, dim=0).double())
            else:
                self.register_buffer('betas', torch.linspace(self.df_model_config['beta_1'], self.df_model_config['beta_T'], self.num_timesteps).double())

            alphas = 1. - self.betas
            alphas_bar = torch.cumprod(alphas, dim=0)
            alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:self.num_timesteps]

            self.alphas_bar = alphas_bar
            self.alphas_bar_prev = alphas_bar_prev
            self.register_buffer('sqrt_alphas_bar',            torch.sqrt(alphas_bar))
            self.register_buffer('sqrt_one_minus_alphas_bar',  torch.sqrt(1. - alphas_bar))
            
            self.register_buffer('sqrt_recip_alphas_bar',      torch.sqrt(1. / alphas_bar))
            self.register_buffer('sqrt_recipm1_alphas_bar',    torch.sqrt(1. / alphas_bar - 1))

            self.register_buffer('posterior_var',              self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
            
            self.register_buffer('posterior_log_var_clipped',  torch.log(torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
            self.register_buffer('posterior_mean_coef1',       torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
            self.register_buffer('posterior_mean_coef2',       torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar)) 


            self.ddim_sampler = DDIMSampler(self)

            self.feed_forward = False

        else:
            self.feed_forward = True
    
    
    def set_input(self, conditions=None, gdf_cond=None):
        self.conditions = conditions.to(self.device)
        self.unconditions = torch.zeros_like(self.conditions).to(self.device)

        # self.gdf_cond = None
        if self.data_type == 'gs':
            self.gdf_cond = gdf_cond.to(self.device)
        else:
            self.gdf_cond = None
        
        self.cond_embeds = self.cond_model.encode(self.conditions).float()
        self.uncond_embeds = self.cond_model.encode(self.unconditions).float()

        if self.cond_model_config.get('adapter', False):
            self.cond_embeds = self.adapter(self.cond_embeds)
            self.uncond_embeds = self.adapter(self.uncond_embeds)

    
    def voxel_sdf_preprocess(self, volume_sdf, size=32):
        maxs, mins = 1, -1
        
        scale_factor = 1
        volume_sdf = (torch.clamp(volume_sdf, min=mins, max=maxs) - mins) * 2 * scale_factor / (maxs - mins) - scale_factor # -1 ~ 1

        volume_sdf = volume_sdf.reshape((size, size, size, -1)).detach()
        volume_sdf = rearrange(volume_sdf, 'h w d c -> c h w d')

        return volume_sdf

    @torch.no_grad()
    def inference(
        self, 
        gdf_cond=None,
        condition=None, 
        ddim_steps=None, 
        ddim_eta=0., 
        uc_scale=None,
        mean_var=True,
    ):
        
        if ddim_steps == None:
            ddim_steps = self.ddim_steps
        
        if uc_scale == None:
            uc_scale = 1.
        
        if isinstance(condition, PIL.Image.Image):
            condition = self.image_preprocess(condition)
            condition = condition.unsqueeze(0).to(self.device)
        elif isinstance(condition, torch.Tensor):
            assert condition.shape[0] == 1

        if isinstance(gdf_cond, torch.Tensor):
            gdf_cond = self.voxel_sdf_preprocess(gdf_cond)
            gdf_cond = gdf_cond.unsqueeze(0).to(self.device)
            

        self.set_input(condition, gdf_cond)
        B = self.conditions.shape[0] # 1 for inference

        shape = [self.unet_config['UNet3D_kwargs']['out_channels']] + self.df_shape
        


        if self.feed_forward:
            samples = self.apply_ff_model(self.cond_embeds)
            return samples, []
        else:
            samples, intermediates = self.ddim_sampler.sample(
                                        S=ddim_steps,
                                        batch_size=B,
                                        shape=shape,
                                        conditioning=self.cond_embeds,
                                        # gdf_cond=self.gdf_cond,
                                        verbose=False,
                                        unconditional_guidance_scale=uc_scale,
                                        unconditional_conditioning=self.uncond_embeds,
                                        eta=ddim_eta,
                                        quantize_x0=False,
                                        use_clamp=True,
                                        clamp_scale=self.clamp_max,
                                        mean_var=mean_var,
                                        scale_factor=self.data_dist_scale,
                                    )

            return samples, intermediates['pred_x0'] # , render_out




    def q_sample(self, x_start, t, noise=None, use_clamp=False):
        if noise == None:
            noise = torch.randn_like(x_start)
            
        ret =  (extract(          self.sqrt_alphas_bar, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_bar, t, x_start.shape) * noise)

        if use_clamp:
            return torch.clamp(ret, self.clamp_min, self.clamp_max)
        else:
            return ret

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.df_model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}
        
        if self.data_type == 'gdf':
            out = self.df_model(x_noisy, t, **cond)
        else:
            x_noisy_cond = torch.cat([x_noisy, self.gdf_cond], dim=1) 
            out = self.df_model(x_noisy_cond, t, **cond)
            
        if isinstance(out, tuple) and not return_ids:
            return out[0]
        else:
            return out
    
    def apply_ff_model(self, cond, return_ids=False):
        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.df_model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}
        
        
        if self.unet_config['model_type'] == "UNet3DGDF":
            out = self.df_model(self.gdf_cond, **cond)
        else:
            out = self.df_model(         None, **cond)

        if isinstance(out, tuple) and not return_ids:
            return out[0]
        else:
            return out
    

    def process_test_x(
        self, 
        x,
    ):
        means = torch.tensor([     
            0.00012763,  0.00000971, -0.00051591,      
            0.08967438, -0.06474344, -0.21201491,     
            -3.92665195, -4.98916960, -5.83290148,      
            1.29501426, -0.56918037,  1.69296813,
            -5.28034639
        ]).to(x.device)

        sqrt_vars = torch.tensor([
            0.02722617, 0.02745687, 0.02772274, 
            0.92372596, 0.93814343, 0.95081377,
            0.68432236, 0.84192544, 0.43621218, 
            0.23838308, 1.99511790, 0.74199206,
            2.05200052
        ]).to(x.device)
        
        x = x * sqrt_vars + means

        rotation_start_idx = 3 + 3 + 3 # offset + rgb + scale
        rotation = x[..., rotation_start_idx:rotation_start_idx+3]

        rotation = threed_to_fourd(rotation)
        x_proc = torch.cat((x[..., :rotation_start_idx], rotation, x[..., rotation_start_idx+3:]), dim=-1)
        
        return x_proc
        

    def display(
        self, 
        x, 
        render_path='.', 
        bg_color=[1, 1, 1], # white bg
    ):
        
        frames = []
        
        x = [i[0] for i in x]
        test_poses = get_test_poses().to(x[0].device)
        
        background = torch.tensor(bg_color, dtype=torch.float32, device=x[0].device)
        x = [rearrange(i, 'c h w d -> h w d c').reshape((32**3, -1)) for i in x]

        for idx in range(len(test_poses)):
            viewpoint_cam = test_poses[idx]
            render_pkg = render(viewpoint_cam, self.process_test_x(x[-1]), background, mean_var=True)
            image = render_pkg["render"]
            frames.append(image)

        imgs = []
        for frame in frames:
            f = frame.clamp(0, 1).detach().permute(1, 2, 0).rot90().cpu().numpy() * 255
            f = f.astype(np.uint8)
            imgs.append(f)

        output_file = os.path.join(render_path, 'sample.gif')
            
        print(f'[INFO] {len(imgs)} frames')
        imageio.mimsave(output_file, imgs, duration=50)
        print(f'[INFO] Saved gif !')
        
        recover_x = self.process_test_x(x[-1])
        return recover_x