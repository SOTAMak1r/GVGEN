import torch
import os
import imageio.v2 as imageio
import cv2
import torch.nn as nn
from collections import defaultdict
import numpy as np
import gc
import json
import PIL
import torchvision
import random
from einops import rearrange
from torch.utils.data import DataLoader
# from basicsr.losses.basic_loss import PerceptualLoss
from utils.loss_utils import ssim, l1_loss, l2_loss, smooth_l1_loss
from gaussian_renderer import render
import math
import pickle
import torch.nn.functional as F
# from datasets import ObjaverseDataset
from models.clip_networks.adapter import CLIPImageEncoderAdapter
from typing import List, Dict
from pytorch_lightning import LightningModule, Trainer
from functools import partial
# from models import get_renderer, compose_featmaps, decompose_featmaps
# from models.render_utils.rays import RayBundle
from models.diffusion_networks.network import DiffusionNet
from models.clip_networks.network import CLIPImageEncoder, CLIPTextEncoder
from models.dino_networks.network import VitExtractor
from models.diffusion_networks.samplers.ddim import DDIMSampler
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision import transforms as pth_transforms
from tqdm import tqdm
from datasets.preprocess import *
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# optimizer, scheduler
from utils import *

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def extract(v, t, x_shape):
    out = torch.gather(v, index=t, dim=0).float()
    return out.contiguous().view([t.shape[0]] + [1] * (len(x_shape) -1))

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def get_test_poses():

    # def get_camera_positions_on_sphere(radius, center_point, num_cameras):
    #     cameras = []
    #     phi = np.pi * (3 - np.sqrt(5))  # Golden angle in radians

    #     for i in range(num_cameras):

    #         x = 1 - (i / float(num_cameras - 1)) * 2  # y goes from 1 to -1
    #         radius_at_height = np.sqrt(1 - x * x)  # Radius at height y

    #         theta = phi * i /  num_cameras * 8 # Golden angle increment

    #         z = np.cos(theta) * radius_at_height
    #         y = np.sin(theta) * radius_at_height

    #         camera_position = np.array([x, y, z]) * radius + np.array(center_point)
    #         cameras.append(camera_position)

    #     return cameras

    def get_camera_positions_on_sphere(radius, center_point, num_cameras):
        cameras = []
        # phi = np.pi * (3 - np.sqrt(5))  # Golden angle in radians

        for i in range(num_cameras):

            # x = 1 - (i / float(num_cameras - 1)) * 2  # y goes from 1 to -1
            x = -0.5
            radius_at_height = np.sqrt(1 - x * x)  # Radius at height y

            # theta = phi * i /  num_cameras * 8 # Golden angle increment
            theta = np.pi * 2 * i /  num_cameras # Golden angle increment

            z = np.cos(theta) * radius_at_height
            y = np.sin(theta) * radius_at_height

            # camera_position = np.array([x, y, z]) * radius + np.array(center_point)
            # camera_position = np.array([y, x, z]) * radius + np.array(center_point)
            # camera_position = np.array([z, y, x]) * radius + np.array(center_point)
            camera_position = np.array([y, z, x]) * radius + np.array(center_point)
            cameras.append(camera_position)

        return cameras

    def get_rotation_matrix_to_look_at(target_point, camera_position):
        forward = target_point - camera_position
        forward = forward / np.linalg.norm(forward)

        # right = np.array([0, 1, 0])  # Assume up direction is positive y-axis
        right = np.array([0, 0, 1])  # Assume up direction is positive z-axis
        right = right - np.dot(right, forward) * forward
        right = right / np.linalg.norm(right)

        up = np.cross(forward, right)

        rotation_matrix = np.column_stack((right, up, -forward))

        return rotation_matrix

    # 设置参数
    R = 0.02  # 球的半径
    center_point = (0, 0, 2)  # 对准点的坐标
    image_num = 60  # 相机数量

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

def get_train_poses():
    train_poses_path = "/mnt/lustre/hexianglong/FINAL_GS_GEN/dm_pipeline_sdf/ckpts/train_cameras.pth"
    return torch.load(train_poses_path)


def get_data_type(in_channels, out_channels):
    if in_channels == 1 and out_channels == 1:
        return 'sdf'
    if in_channels == 8 and out_channels == 1:
        return 'sdf'
    return 'gs'

# def _get_grid(voxel_size, world_size):
#     N = voxel_size
#     interval = N - 1
#     grid_unseen_xyz = torch.zeros((N, N, N, 3))
#     for i in range(N):
#         grid_unseen_xyz[i, :, :, 0] = i
#     for j in range(N):
#         grid_unseen_xyz[:, j, :, 1] = j
#     for k in range(N):
#         grid_unseen_xyz[:, :, k, 2] = k
#     grid_unseen_xyz -= (interval / 2.0)
#     grid_unseen_xyz /= (interval / 2.0) / world_size
#     return grid_unseen_xyz

class ImageConditioned_DiffusionModel(LightningModule):
    def __init__(
        self, 
        opts,
        cfg,
        mode='train',
        # volume_type: grid triplane
    ):
        super(ImageConditioned_DiffusionModel, self).__init__()
        self.validation_step_outputs = []

        # init configs
        # pcplayer_weight = {
        #     'conv1_2': 0,
        #     'conv2_2': 0,
        #     'conv3_4': 1,
        #     'conv4_4': 1,
        #     'conv5_4': 1
        # }
        # self.cri_perceptual = PerceptualLoss(layer_weights=pcplayer_weight, vgg_type='vgg19').to(self.device)
        # self.xyz_volume = rearrange(_get_grid(32, world_size=0.5 * 10.), 'h w d c -> c h w d')[None] #.repeat(1, 1, 1, 1) # TODO: BATCH_SIZE

        ##################################################################
        # self.clamp_min, self.clamp_max = -16, 16
        # self.data_dist_scale = 5.
        self.clamp_min, self.clamp_max = -5, 5
        self.data_dist_scale = 1.
        ##################################################################

        self.val_num = 0
        self.mode = mode
        self.opts = opts
        self.cond_model_config = cfg['cond_model_config']
        self.volume_type = cfg['volume_type']
        self.unet_config = cfg['unet_config']
        self.df_model_config = cfg['df_model_config']
        self.render_config = cfg['render_config']
        self.use_render_loss = self.render_config.get('use_render_loss', False)

        self.ddim_steps = 100
        self.drop_rate = self.df_model_config['drop_rate']
        self.trainable_models = []
        self.df_shape = self.df_model_config['df_shape']

        # init condition model
        cond_model_type = self.cond_model_config.get('model_type', 'clip')
        cond_mode = self.cond_model_config.get('cond_mode', 'image')
        print(f'[INFO] condition model is {cond_model_type, cond_mode}')

        self.cond_model_type = cond_model_type
        self.cond_mode = cond_mode

        if cond_mode == 'image':
            self.cond_model = CLIPImageEncoder(model=self.cond_model_config['model'])
            self.cond_model.to(self.device)
        else:
            self.cond_model = CLIPTextEncoder(model=self.cond_model_config['model'])
            self.cond_model.to(self.device)

        if self.mode == 'train' and self.cond_model_config['finetune'] == False:
            for param in self.cond_model.parameters():
                param.requires_grad = False
                
        self.data_type = get_data_type(self.unet_config['UNet3D_kwargs']['in_channels'], self.unet_config['UNet3D_kwargs']['out_channels'])
        print(f'[INFO] Diffusion data type is {self.data_type}')

        # init clip adapter
        if self.cond_model_config.get('adapter', False):
            print(f'[INFO] ADAPTER involve !!!')
            self.adapter = CLIPImageEncoderAdapter(c_in=self.unet_config['UNet3D_kwargs']['context_dim'])
            self.adapter.to(self.device)
            if self.mode == 'train':
                self.trainable_models += [self.adapter]

        # init diffusion backbone
        self.df_model = DiffusionNet(self.unet_config, conditioning_key=self.df_model_config['conditioning_key'])
        self.df_model.to(self.device)

        if self.mode == 'train':
            self.trainable_models += [self.df_model]
            self.train_dataset = ObjaverseDataset(
                                    base_path=opts.dataset_dir, 
                                    mode='train', 
                                    multi_img_path=opts.multi_img_path, 
                                    cond_model_type=self.cond_model_type, 
                                    data_dist_scale=self.data_dist_scale,
                                    data_type = self.data_type,
                                    use_render_loss = self.use_render_loss,
                                    cond_mode = self.cond_mode,
                                    # use_loss_weight=self.use_loss_weight,
                                )
            self.val_dataset = ObjaverseDataset(
                                    base_path=opts.dataset_dir, 
                                    mode='val', 
                                    multi_img_path=opts.multi_img_path, 
                                    cond_model_type=self.cond_model_type, 
                                    data_dist_scale=self.data_dist_scale,
                                    data_type = self.data_type,
                                    use_render_loss = self.use_render_loss,
                                    cond_mode = self.cond_mode,
                                    # use_loss_weight=self.use_loss_weight,
                                )
        self.temp_render_path = opts.temp_render_path

        ##################################################################################################################
        self.l_simple_weight = self.df_model_config['l_simple_weight']
        
        # print(f'[DEBUG] CNM {self.unet_config}')
        if self.unet_config['model_type'] != "UNet3DSDF" and self.unet_config['model_type'] != "UNet3DLRM":
            self.num_timesteps = self.df_model_config['T']

            #self.optim_lr = self.df_model_config['optim_lr']
            #self.grad_clip = self.df_model_config['grad_clip']
            #self.var_type = self.df_model_config['var_type']
            #self.mean_type = self.df_model_config['mean_type']
            
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
            
            # calculations for diffusion q(x_t | x_{t-1}) and others
            self.register_buffer('sqrt_recip_alphas_bar',      torch.sqrt(1. / alphas_bar))
            self.register_buffer('sqrt_recipm1_alphas_bar',    torch.sqrt(1. / alphas_bar - 1))

            # calculations for posterior q(x_{t-1} | x_t, x_0)
            self.register_buffer('posterior_var',              self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
            
            # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
            self.register_buffer('posterior_log_var_clipped',  torch.log(torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
            self.register_buffer('posterior_mean_coef1',       torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
            self.register_buffer('posterior_mean_coef2',       torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar)) 
            to_torch = partial(torch.tensor, dtype=torch.float32)
            
            if self.parameterization == "eps":
                lvlb_weights = self.betas ** 2 / (2 * self.posterior_var * to_torch(alphas).to(self.device) * (1 - self.alphas_bar))
            elif self.parameterization == "x0":
                lvlb_weights = 0.5 * torch.sqrt(alphas_bar) / (2. * 1 - alphas_bar)
            else:
                raise NotImplementedError(f"{self.parameterization} parameterization not supported!")
            
            # TODO how to choose this term
            lvlb_weights[0] = lvlb_weights[1]
            self.lvlb_weights = lvlb_weights

            logvar_init = 0.
            self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,)).to(self.device)
            
            assert not torch.isnan(self.lvlb_weights).all()

            self.ddim_sampler = DDIMSampler(self)

            self.feed_forward = False

        else:
            print(F'[INFO] FEED FORWARD YYDS !!!! ')
            # self.register_buffer('feed_forward', True)
            self.feed_forward = True
    
    def init_weight(self, pt_model=None):

        if pt_model != None:

            # print(f'[INFO] torch before {torch.cuda.memory_allocated(0), torch.cuda.memory_cached(0)}')
            gc.collect()
            torch.cuda.empty_cache()
            # print(f'[INFO] torch before {torch.cuda.memory_allocated(0), torch.cuda.memory_cached(0)}')


            if pt_model == 'DAME': # means DIY !!!!
                print(f'[DAME] !!!!!!!!!')
                
                prior_pt = "/mnt/lustre/hexianglong/FINAL_GS_GEN/dm_pipeline_sdf/ckpts/BASELINE_sdf/last-v2.ckpt"
                dic = torch.load(prior_pt, map_location='cpu')

                new_model_dict = self.state_dict()

                selected_params_dir = '/mnt/lustre/hexianglong/FINAL_GS_GEN/dm_pipeline_sdf/ckpts/param_select_nocond.json'
                with open(selected_params_dir, 'r') as f:
                    selected_params = json.load(f)
                selected_params = selected_params['param_name']

                cnt_para = 0
                for param_name, param in dic['state_dict'].items():
                    if param_name in selected_params:
                        new_model_dict[param_name] = param
                        cnt_para += 1

                self.load_state_dict(new_model_dict, strict=False)
                print(f"[INFO] Loaded model from <{prior_pt}>, cnt_para = {cnt_para}")

            else:
                dic = torch.load(pt_model, map_location='cpu')
                self.load_state_dict(dic['state_dict'], strict=False)
                print(f"[INFO] Loaded model from <{pt_model}>")

    
    def set_input(self, cond_images=None, sdf_cond=None):
        self.cond_images = cond_images.to(self.device)
        self.uncond_images = torch.zeros_like(self.cond_images).to(self.device)

        # self.sdf_cond = None
        if self.data_type == 'gs':
            self.sdf_cond = sdf_cond.to(self.device)
        else:
            self.sdf_cond = None
        
        if self.mode == 'train' or self.drop_rate == 1.0:
            mask = torch.rand(self.cond_images.shape[0])
            self.cond_images[mask <= self.drop_rate] *= 0
            # if self.data_type == 'gs':
            #     self.sdf_cond[mask <= self.drop_rate] *= 0

        self.cond_embeds = self.cond_model.encode(self.cond_images).float()
        self.uncond_embeds = self.cond_model.encode(self.uncond_images).float()

        if self.cond_model_config.get('adapter', False):
            self.cond_embeds = self.adapter(self.cond_embeds)
            self.uncond_embeds = self.adapter(self.uncond_embeds)

    def forward(self):
        pass

    def image_preprocess(self, img, img_px=512):
        # just for data aug, clip_preprocess is executed in clip_model automatically.
        if self.cond_model_type == 'clip':
            transform = Compose([
                Resize((img_px, img_px), interpolation=BICUBIC),
                _convert_image_to_rgb,
                ToTensor(),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        else:
            print(f'[INFO] DINO !!!')
            transform = pth_transforms.Compose([
                pth_transforms.Resize((512, 512)),
                pth_transforms.ToTensor(),
                pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        return transform(img)
    
    def voxel_sdf_preprocess(self, volume_sdf, from_generate=False, size=32):
        if from_generate:
            maxs, mins = 1, -1
        else:
            maxs, mins = 0.15, 0
        
        scale_factor = 1
        volume_sdf = (torch.clamp(volume_sdf, min=mins, max=maxs) - mins) * 2 * scale_factor / (maxs - mins) - scale_factor # -1 ~ 1

        # print(f'[INFO] cond sdf shape = {volume_sdf.shape, volume_sdf.max().item(), volume_sdf.min().item()}')
        volume_sdf = volume_sdf.reshape((size, size, size, -1)).detach()
        volume_sdf = rearrange(volume_sdf, 'h w d c -> c h w d')

        return volume_sdf

    # @torch.no_grad()
    def inference(
        self, 
        sdf_cond=None,
        sdf_from_generate=False,
        cond_image=None, 
        ddim_steps=None, 
        ddim_eta=0., 
        uc_scale=None,
        infer_all=False, 
        ray_o=None, 
        ray_d=None,
        use_clamp=True,
        mean_var=False,
        render_poses=None,
    ):
    #     '''
    #     TODO: adapt triplane while too big
    #     cond_image: torch.Tensor
    #     '''
        # ddim_steps = 500
        if ddim_steps == None:
            ddim_steps = self.ddim_steps
        
        if uc_scale == None:
            uc_scale = 1. #self.uc_scale
        
        if isinstance(cond_image, PIL.Image.Image):
            cond_image = self.image_preprocess(cond_image)
            cond_image = cond_image.unsqueeze(0).to(self.device)
        elif isinstance(cond_image, torch.Tensor):
            assert cond_image.shape[0] == 1

        if isinstance(sdf_cond, torch.Tensor):
            sdf_cond = self.voxel_sdf_preprocess(sdf_cond, from_generate=sdf_from_generate)
            sdf_cond = sdf_cond.unsqueeze(0).to(self.device)
        # cond_image *= 0. # DEBUG

        self.set_input(cond_image, sdf_cond)
        B = self.cond_images.shape[0] # 1 for inference

        shape = [self.unet_config['UNet3D_kwargs']['out_channels']] + self.df_shape
        # print(f'[DEBUG] Diffusion data shape = {shape}')

        self.render_poses = render_poses

        if self.feed_forward:
            samples = self.apply_ff_model(self.cond_embeds)
            return samples, []
        else:
            samples, intermediates = self.ddim_sampler.sample(
                                        S=ddim_steps,
                                        batch_size=B,
                                        shape=shape,
                                        conditioning=self.cond_embeds,
                                        sdf_cond=self.sdf_cond,
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

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=16,
                          batch_size=self.opts.batch_size,
                          pin_memory=False)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=8,
                          batch_size=1,
                          pin_memory=False)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.opts, self.trainable_models)
        self.scheduler = get_scheduler(self.opts, self.optimizer)
        return [self.optimizer], [self.scheduler]
    
    def unpack_batch(self, batch):
        volumes = batch['volume'].to(self.device)
        cond_images = batch['cond_image'].to(self.device)

        volumes_sdf = batch['sdf'].to(self.device) if self.data_type == 'gs' else None

        render_images, render_poses = None, None
        if self.use_render_loss:
            render_images = batch['render_images'].to(self.device)
            render_poses = batch['render_poses'].to(self.device)
        
        return volumes, cond_images, volumes_sdf, render_images, render_poses

    def training_step(self, batch, batch_idx):
        # print(f'[INFO] lr = {get_learning_rate(self.optimizer)}')
        self.mode = 'train'
        self.log('lr', get_learning_rate(self.optimizer), prog_bar=True)

        loss, render_loss = self.share_step(batch, batch_idx) 

        self.log('train_loss', loss, prog_bar=True)
        # if self.current_epoch >= self.render_config['render_loss_start_epoch']:
        #     self.log('train_render_loss', render_loss, prog_bar=True) 
        # else:
        #     self.log('train_render_loss', render_loss, prog_bar=False) 
        if self.use_render_loss:
            self.log('train_render_loss', render_loss, prog_bar=True) 

        return {'loss' : loss + render_loss}
    
    def validation_step(self, batch, batch_idx):
        self.optimizer.zero_grad()
        # self.scheduler['scheduler'].zero_grad()

        self.mode = 'val'
        loss, render_loss = self.share_step(batch, batch_idx)
        
        self.log('val_loss', loss, prog_bar=False)
        self.log('val_render_loss', render_loss, prog_bar=False)
        self.validation_step_outputs.append({'loss': loss, 'render_loss': render_loss})

        return {'loss': loss, 'render_loss': render_loss}

    def share_step(self, batch, batch_idx):
        volumes, cond_images, volume_sdf, render_images, render_poses = self.unpack_batch(batch)
        # print("volumes shape:", volumes.shape)
        self.set_input(cond_images, volume_sdf)
        if self.feed_forward:
            t = None
        else:
            t = torch.randint(0, self.num_timesteps, size=(volumes.shape[0],), device=self.device).long()
        # z_noisy, target, loss, render_loss = self.p_losses(volumes, self.cond_embeds, t, render_images, render_poses)
        target, loss, render_loss = self.p_losses(volumes, self.cond_embeds, t, render_images, render_poses)
        return loss, render_loss
        
    def on_validation_epoch_end(self):
        val_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        val_render_loss = torch.stack([x['render_loss'] for x in self.validation_step_outputs]).mean()

        self.log('val_loss', val_loss, prog_bar=False)
        self.log('val_render_loss', val_render_loss, prog_bar=False)

        self.val_num += 1

        print(f'[Val INFO] val_loss = {val_loss}, val_render_loss = {val_render_loss}, total_loss = {val_loss + val_render_loss}')
        self.validation_step_outputs.clear()

    def q_sample(self, x_start, t, noise=None, use_clamp=False):
        if noise == None:
            noise = torch.randn_like(x_start)
        # if self.opts.debug:
        #     print(extract(          self.sqrt_alphas_bar, t, x_start.shape))
        #     print(extract(self.sqrt_one_minus_alphas_bar, t, x_start.shape))
        ret =  (extract(          self.sqrt_alphas_bar, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_bar, t, x_start.shape) * noise)
        
        # print(f'\n[INFO] ret = {ret.max().item(), ret.min().item(), use_clamp}')

        if use_clamp:
            return torch.clamp(ret, self.clamp_min, self.clamp_max)
        else:
            return ret

        # return ret


    # def apply_x0_backward(self, x0, bg_color=[1, 1, 1]):
    #     background = torch.tensor(bg_color, dtype=torch.float32, device=x0.device)

    #     # y = x0.clone()#.requires_grad_(True)
    #     y = nn.Parameter(torch.tensor(x0.clone(), requires_grad=True))
    #     print(f'[DEBUG] y = {y.requires_grad, y.device, x0.device}')

    #     y = rearrange(y, 'b c h w d -> b h w d c').contiguous().requires_grad_(True)
    #     # y = y.permute(0, 4, 1, 2, 3)
    #     print(f'[DEBUG] y1 = {y.requires_grad, y.device, x0.device}')
    #     y = y.reshape((y.shape[0], 32**3, -1)).contiguous().requires_grad_(True)
    #     print(f'[DEBUG] y2 = {y.requires_grad, y.device, x0.device}')
    #     # rotation_one = torch.ones((pcs.shape[0], 32**3, 1)).to(device=pcs.device)
    #     # rotation_start_idx = 3 + 3 + 3 # offset + rgb + scale
    #     # pcs = torch.cat((pcs[..., :rotation_start_idx], rotation_one, pcs[..., rotation_start_idx:]), dim=-1).contiguous()

    #     # for batch_idx in range(len(output)):
    #     #     for i in range(len(render_images[batch_idx])):

    #             # print(f'[DEBUG] render_poses = {batch_idx, i, render_poses.shape, render_images.shape}')
    #     # print(f'[DEBUG] render_poses = {type(self.render_poses), self.render_poses.shape}')
    #     i = random.sample(range(len(self.render_poses)), 1)
        
    #     viewpoint_cam = self.render_poses[i].to(x0.device)[0]
    #     # print(f'[DEBUG] i = {i, viewpoint_cam.shape}')

    #     # y[0] = y[0].requires_grad_(True)
    #     # print(f'[DEBUG] y leaf = {y.is_leaf, y[0].is_leaf}')
    #     # print(f'[DEBUG] y = {y.requires_grad, y[0].requires_grad}')

    #     render_pkg = render(viewpoint_cam, self.process_test_x(y[0], mean_var=True, polar=True), background, mean_var=True)

    #     # render_pkg = render(viewpoint_cam, pcs[batch_idx], background, data_dist_scale=self.data_dist_scale)
    #     image = render_pkg["render"]
    #     print(f'[DEBUG] image = {image.shape, image.max(), image.min(), image.requires_grad}')
        
    #     gt_image = self.cond_embeds
    #     print(f'[DEBUG] gt_image = {gt_image.shape, gt_image.max(), gt_image.min()}')
    #     exit(0)



    #     if batch_idx == 0 and i == 0:
    #         torchvision.utils.save_image(image, os.path.join(self.temp_render_path, '{0:05d}'.format(batch_idx) + '_' +  str(i) + "_render.png"))
    #         torchvision.utils.save_image(gt_image, os.path.join(self.temp_render_path, '{0:05d}'.format(batch_idx) + '_' +  str(i) + "_gt.png"))

    #     if self.render_config['l1_weight'] > 0.:
    #         accu_l1_loss = accu_l1_loss + l1_loss(image, gt_image)
    #     if self.render_config['ssim_weight'] > 0.:
    #         accu_ssim_loss = accu_ssim_loss + (1.0 - ssim(image, gt_image))
                    



    #     return x0

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.df_model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}
        
        if self.data_type == 'sdf':
            out = self.df_model(x_noisy, t, **cond)
        else:
            x_noisy_cond = torch.cat([x_noisy, self.sdf_cond], dim=1) 
            out = self.df_model(x_noisy_cond, t, **cond)
            # print(f'[INFO] out = {out.shape, out.requires_grad}')
        # out = self.df_model(x_noisy, t, **cond)

        # if self.mode == 'test':
        #     if isinstance(self.render_poses, torch.Tensor): ###################### HERE
        #         out = self.apply_x0_backward(out)
        
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
        
        # print(f'[DEBUG] self.sdf_cond = {self.sdf_cond}')
        if self.unet_config['model_type'] == "UNet3DSDF":
            out = self.df_model(self.sdf_cond, **cond)
        else:
            out = self.df_model(         None, **cond)

        if isinstance(out, tuple) and not return_ids:
            return out[0]
        else:
            return out
    


    def get_loss(self, pred, target, loss_type='l2', mean=True):
        if loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def gaussian_render(self, output, render_images, render_poses, bg_color=[1, 1, 1]):
        
        background = torch.tensor(bg_color, dtype=torch.float32, device=render_images.device)

        accu_ssim_loss = 0.
        accu_l1_loss   = 0.

        pcs = rearrange(output, 'b c h w d -> b h w d c').contiguous()
        pcs = pcs.reshape((pcs.shape[0], 32**3, -1)).contiguous()
        # rotation_one = torch.ones((pcs.shape[0], 32**3, 1)).to(device=pcs.device)
        # rotation_start_idx = 3 + 3 + 3 # offset + rgb + scale
        # pcs = torch.cat((pcs[..., :rotation_start_idx], rotation_one, pcs[..., rotation_start_idx:]), dim=-1).contiguous()

        for batch_idx in range(len(output)):
            for i in range(len(render_images[batch_idx])):
                # print(f'[DEBUG] render_poses = {batch_idx, i, render_poses.shape, render_images.shape}')
                viewpoint_cam = render_poses[batch_idx, i]

                render_pkg = render(viewpoint_cam, self.process_test_x(pcs[batch_idx], mean_var=True, polar=True), background, mean_var=True)

                # render_pkg = render(viewpoint_cam, pcs[batch_idx], background, data_dist_scale=self.data_dist_scale)
                image = render_pkg["render"]
                gt_image = render_images[batch_idx, i]

                if batch_idx == 0 and i == 0:
                    torchvision.utils.save_image(image, os.path.join(self.temp_render_path, '{0:05d}'.format(self.current_epoch) + '_' +  str(i) + "_render.png"))
                    torchvision.utils.save_image(gt_image, os.path.join(self.temp_render_path, '{0:05d}'.format(self.current_epoch) + '_' +  str(i) + "_gt.png"))

                if self.render_config['l1_weight'] > 0.:
                    accu_l1_loss = accu_l1_loss + l1_loss(image, gt_image)
                if self.render_config['ssim_weight'] > 0.:
                    accu_ssim_loss = accu_ssim_loss + (1.0 - ssim(image, gt_image))
                    

        # 80 0.08 1.5
        # 0.4 0.016 1.2
        loss = (
              accu_l1_loss    * self.render_config['l1_weight']   \
            + accu_ssim_loss  * self.render_config['ssim_weight'] \
        ) / (len(render_images))

        return loss
    
    # def process_test_x(self, x):
    #     rotation_one = torch.ones((32**3, 1)).to(device=x.device)
    #     rotation_start_idx = 3 + 3 + 3 # offset + rgb + scale
    #     x_proc = torch.cat((x[..., :rotation_start_idx], rotation_one, x[..., rotation_start_idx:]), dim=-1)#.contiguous()
    #     return x_proc
    
    def process_test_x(self, x, debug_gt=False, mean_var=False, polar=False):
        # print(f'[DEBUG] y top = {x.requires_grad}')
        if mean_var:
            # means = torch.tensor([     
            #     0.00012763,  0.00000971, -0.00051591,
            #     0.08967438, -0.06474344, -0.21201491,     
            #     -3.92665195, -4.98916960, -5.83290148,      
            #     -0.09364603, -0.10800153, -0.09485082,     
            #     2.73539782
            # ]).to(x.device)

            # sqrt_vars  = torch.tensor([
            #     0.02722617, 0.02745687, 0.02772274, 
            #     0.92372596, 0.93814343, 0.95081377,
            #     0.68432236, 0.84192544, 0.43621218, 
            #     0.61441237, 0.30088466, 0.61463439, 
            #     3.62624121
            # ]).to(x.device)
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
            
            # print(f'\n[INFO] x 1 = {x[..., :3].max(), x[..., :3].min()}')
            x = x * sqrt_vars + means
            # print(f'[INFO] x 2 = {x[..., :3].max(), x[..., :3].min()}')

        if debug_gt:
            return x

        rotation_start_idx = 3 + 3 + 3 # offset + rgb + scale
        rotation = x[..., rotation_start_idx:rotation_start_idx+3]

        if polar:
            # print(f'[DEBUG] rotation 1 = {rotation.requires_grad}')
            rotation = threed_to_fourd(rotation)
            # print(f'[DEBUG] rotation 2 = {rotation.requires_grad}')
            x_proc = torch.cat((x[..., :rotation_start_idx], rotation, x[..., rotation_start_idx+3:]), dim=-1)#.requires_grad_(True)
        else:
            rotation = torch.clamp(rotation, min=-self.data_dist_scale, max=self.data_dist_scale)
            rotation_w = torch.sqrt(torch.clamp(1 - torch.pow(rotation[..., 0], 2) + torch.pow(rotation[..., 1], 2) + torch.pow(rotation[..., 2], 2), min=0))
            # print(f'[DEBUG] {rotation_w.shape, x.shape}')
            x_proc = torch.cat((x[..., :rotation_start_idx], rotation_w.unsqueeze(-1), x[..., rotation_start_idx:]), dim=-1)#.contiguous()

        # print(f'[DEBUG] x_proc = {x_proc.requires_grad}')
        return x_proc
        
    def display(
        self, 
        x, 
        render_path='/mnt/hwfile/3dv/chenjunyi/2024_2_ECCV/GVGEN/results', 
        bg_color=[1, 1, 1],
        gt_x=None,
        debug_gt=False,
        mean_var=False,
        polar=False,
        only_recover=False,
        obj_name=None,
        show_process=False,
        save_obj_path='objs'
    ):
        # only 1 sample input [1, C, H, W, D]
        frame_path = os.path.join(render_path, 'frames')
        frames = []
        
        with torch.no_grad():
            cnt = 0
            x = [i[0] for i in x]
            test_poses = get_test_poses().to(x[0].device)
            # test_poses = get_train_poses().to(x[0].device)
            
            background = torch.tensor(bg_color, dtype=torch.float32, device=x[0].device)
            x = [rearrange(i, 'c h w d -> h w d c').reshape((32**3, -1)) for i in x]

            # if only_recover:
            #     recover_x = self.process_test_x(x[-1], mean_var=mean_var, polar=polar)
            #     return recover_x
                
            if not self.feed_forward or show_process:
                for idx in range(len(x)-1):
                    viewpoint_cam = test_poses[0]
                    render_pkg = render(viewpoint_cam, self.process_test_x(x[idx], mean_var=mean_var, polar=polar), background, mean_var=mean_var)
                    image = render_pkg["render"]
                    # torchvision.utils.save_image(image, os.path.join(frame_path, '{0:03d}'.format(cnt) + "_test.png"))
                    cnt += 1

            for idx in range(len(test_poses)):
                viewpoint_cam = test_poses[idx]
                render_pkg = render(viewpoint_cam, self.process_test_x(x[-1], mean_var=mean_var, polar=polar), background, mean_var=mean_var)
                image = render_pkg["render"]
                # print(f'[INFO] image = {type(image)}')
                # print(f'[INFO] image = {image.shape}')
                frames.append(image)
                # torchvision.utils.save_image(image, os.path.join(frame_path, '{0:03d}'.format(cnt) + "_test.png"))
                cnt += 1
            
            if gt_x != None:
                gt_x = gt_x.to(x[-1].device)
                for idx in range(len(test_poses)):
                    viewpoint_cam = test_poses[idx]
                    render_pkg = render(viewpoint_cam, self.process_test_x(gt_x, debug_gt=debug_gt, mean_var=mean_var, polar=polar), background, mean_var=mean_var)
                    image = render_pkg["render"]
                    # torchvision.utils.save_image(image, os.path.join(frame_path, '{0:03d}'.format(cnt) + "_test.png"))
                    cnt += 1


            # frames = sorted(os.listdir(frame_path))
            imgs = []
            # for frame in frames[:cnt]:
            for frame in frames:
                # imgs.append(imageio.imread(os.path.join(frame_path, frame)))

                # f = frame.clamp(0, 1).detach().permute(1, 2, 0).cpu().numpy() * 255
                f = frame.clamp(0, 1).detach().permute(1, 2, 0).rot90().cpu().numpy() * 255
                f = f.astype(np.uint8)
                imgs.append(f)
                # print(f'[INFO] imgs = {type(imgs[-1])}')
                # print(f'[INFO] imgs = {imgs[-1].shape}')
                
            if obj_name == None:
                output_file = os.path.join(render_path, 'output_ff.gif' if self.feed_forward else 'output_dm.gif')
            else:
                output_file = os.path.join(render_path, save_obj_path, obj_name + '.gif')
                
            try:
                imageio.mimsave(output_file, imgs, duration=50)
                print(f'[INFO] Saved gif!')
            except:
                pass
            

            recover_x = self.process_test_x(x[-1], mean_var=mean_var, polar=polar)
            return recover_x

    def p_losses(self, x_start, cond, t, render_images=None, render_poses=None, noise=None):
        
        if self.feed_forward:
            model_output = self.apply_ff_model(cond)
            target = x_start
        else:
            if noise == None:
                noise = torch.randn_like(x_start)
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
            model_output = self.apply_model(x_noisy, t, cond)

            if self.parameterization == "x0":
                target = x_start
            elif self.parameterization == "eps":
                target = noise
            else:
                raise NotImplementedError(f'{self.parameterization} parameterization not supported!')
            
        loss_simple = self.get_loss(model_output, target, mean=False)
        
        # if self.volume_type == 'grid':
        #     loss_simple = loss_simple.mean([1, 2, 3, 4]) # B, C, 32, 32, 32
        # elif self.volume_type == 'triplane':
        #     loss_simple = loss_simple.mean([1, 2, 3])
        # else:
        #     raise NotImplementedError(f'{self.volume_type} volume type not supported!')

        



        # mse_loss_weights = (
        #                 torch.stack([snr, 5. * torch.ones_like(t)], dim=1).min(dim=1)[0] / snr
        #             )

        # logvar_t = self.logvar[t].to(self.device)
        # loss = loss_simple / torch.exp(logvar_t) + logvar_t
        #if self.learn_logvar:
        #    loss_dict.update({f'loss_gamma': loss.mean()})
        #    loss_dict.update({'logvar': self.logvar.data.mean()})

        
        render_loss = torch.tensor(0.).to(loss_simple.device)
        mse_loss_weights = 1.
        
        if self.use_render_loss:
            if self.feed_forward:
                    render_loss = self.gaussian_render(model_output, render_images, render_poses) * self.render_config['render_weight'] # * (1. - t / 1000.)

            else:
                if self.parameterization == "x0":
                    render_loss = self.gaussian_render(model_output, render_images, render_poses) * self.render_config['render_weight'] # * (1. - t / 1000.)
                else:
                    sqrt_a_t = extract(self.sqrt_alphas_bar, t, x_start.shape) 
                    sqrt_one_minus_at = extract(self.sqrt_one_minus_alphas_bar, t, x_start.shape) 
                    render_loss = self.gaussian_render((x_noisy - sqrt_one_minus_at * model_output) / sqrt_a_t, render_images, render_poses) * 0.8 # * (1. - t / 1000.)
                
            render_loss = render_loss.mean()
            # mse_loss_weights = 0.6
            mse_loss_weights = 1 - self.render_config['render_weight']
            # mse_loss_weights = mse_loss_weights[:, None, None, None, None]
            #print(t.shape, x_start.shape, mse_loss_weights.shape, "mse")
            
        
        loss = loss_simple * mse_loss_weights
        loss = self.l_simple_weight * loss.mean() 
        
        #loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3, 4))
        #loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        #loss_dict.update({f'loss_vlb': loss_vlb})
        #loss += (self.original_elbo_weight * loss_vlb)
        #loss_dict.update({f'loss_total': loss.clone().detach().mean()})
        # return x_noisy, target, loss, render_loss
        return target, loss, render_loss


    
    # def p_losses(self, x_start, cond, t, render_images=None, render_poses=None, noise=None, loss_weight=None):
    #     if noise == None:
    #         noise = torch.randn_like(x_start)
    #     x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

    #     # print(f'[DEBUG] {x_noisy.shape, t.shape, cond.shape}')
    #     model_output = self.apply_model(x_noisy, t, cond)

    #     if self.parameterization == "x0":
    #         target = x_start
    #     elif self.parameterization == "eps":
    #         target = noise
    #     else:
    #         raise NotImplementedError(f'{self.parameterization} parameterization not supported!')

    #     loss_simple = self.get_loss(model_output, target, mean=False)

    #     # print(f'[DEBUG] loss {loss_simple.shape, loss_weight.shape}')
    #     if loss_weight is not None:
    #         loss_simple = loss_simple * loss_weight
        
    #     if self.volume_type == 'grid':
    #         loss_simple = loss_simple.mean([1, 2, 3, 4]) # B, C, 32, 32, 32
    #     elif self.volume_type == 'triplane':
    #         loss_simple = loss_simple.mean([1, 2, 3])
    #     else:
    #         raise NotImplementedError(f'{self.volume_type} volume type not supported!')
        
    #     # print(f'[DEBUG] {t.device, self.logvar.device}')
    #     self.logvar = self.logvar.to(self.device)
    #     logvar_t = self.logvar[t].to(self.device)
    #     loss = loss_simple / torch.exp(logvar_t) + logvar_t

    #     render_loss = torch.tensor(0.).to(loss_simple.device)
        
    #     #if self.learn_logvar:
    #     #    loss_dict.update({f'loss_gamma': loss.mean()})
    #     #    loss_dict.update({'logvar': self.logvar.data.mean()})

    #     loss = self.l_simple_weight * loss.mean()

    #     #loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3, 4))
    #     #loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
    #     #loss_dict.update({f'loss_vlb': loss_vlb})
    #     #loss += (self.original_elbo_weight * loss_vlb)
    #     #loss_dict.update({f'loss_total': loss.clone().detach().mean()})
    #     return x_noisy, target, loss, render_loss

