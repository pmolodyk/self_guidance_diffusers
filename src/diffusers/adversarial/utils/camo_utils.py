import sys
import os
import numpy as np
from easydict import EasyDict
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    TexturesUV, look_at_view_transform, FoVPerspectiveCameras,
    AmbientLights, DirectionalLights, PointLights
)

# add path for demo utils functions 
sys.path.append(os.path.abspath(''))

from . import pytorch3d_modify_utils as p3dmd
from . import mesh_utils as MU


import torch
import itertools
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
import fnmatch
from PIL import Image


class ColorTransform(nn.Module):
    def __init__(self, para_path):
        super(ColorTransform, self).__init__()
        file = np.load(para_path, allow_pickle=True)
        self.degree = file['d']
        weight = torch.from_numpy(file['weight'])
        bias = torch.from_numpy(file['bias'])
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)

    def poly_feature(self, x, degree=None):
        if degree is None:
            degree = self.degree
        n = x.shape[1]
        feature = [x.clone()]
        index = list(range(n))
        for d in range(1, degree):
            new = []
            k = 0
            for i in range(n):
                new.append(x[:, i:i + 1] * feature[-1][:, index[i]:])
                index[i] = k
                k = k + new[-1].shape[1]
            new = torch.cat(new, 1)
            feature.append(new)
        feature = torch.cat(feature, 1)
        return feature

    def forward(self, x):
        f = self.poly_feature(x)
        f = f.transpose(1, -1)
        #     pred = (f.unsqueeze(1) * weight.unsqueeze(0)).sum(2) + bias
        pred = torch.matmul(f, self.weight) + self.bias
        pred = pred.transpose(1, -1)
        return pred


def grid_sample(input, grid, canvas = None):
    output = F.grid_sample(input, grid)
    if canvas is None:
        return output
    else:
        input_mask = Variable(input.data.new(input.size()).fill_(1))
        output_mask = F.grid_sample(input_mask, grid)
        padded_output = output * output_mask + canvas * (1 - output_mask)
        return padded_output


class InriaDataset(Dataset):
    """InriaDataset: representation of the INRIA person dataset.

    Internal representation of the commonly used INRIA person dataset.
    Available at: http://pascal.inrialpes.fr/data/human/

    Attributes:
        len: An integer number of elements in the
        img_dir: Directory containing the images of the INRIA dataset.
        lab_dir: Directory containing the labels of the INRIA dataset.
        img_names: List of all image file names in img_dir.
        shuffle: Whether or not to shuffle the dataset.

    """

    def __init__(self, img_dir, imgsize, shuffle=True, if_square=True):
        n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images = n_png_images + n_jpg_images
        # n_labels = len(fnmatch.filter(os.listdir(lab_dir), '*.txt'))
        # assert n_images == n_labels, "Number of images and number of labels don't match"
        self.len = n_images
        self.img_dir = img_dir
        # self.lab_dir = lab_dir
        self.imgsize = imgsize
        self.img_names = fnmatch.filter(os.listdir(img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
        self.shuffle = shuffle
        self.img_paths = []
        self.if_square = if_square
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        # lab_path = os.path.join(self.lab_dir, self.img_names[idx]).replace('.jpg', '.txt').replace('.png', '.txt')
        image = Image.open(img_path).convert('RGB')


        image = self.pad_and_scale(image)
        transform = transforms.ToTensor()
        image = transform(image)
        # label = self.pad_lab(label)
        return image

    def pad_and_scale(self, img):
        """

        Args:
            img:

        Returns:

        """
        w, h = img.size
        if w==h:
            padded_img = img
        elif self.if_square:
            a = min(w, h)
            ww = (w - a) // 2
            hh = (h - a) // 2
            padded_img = img.crop([ww, hh, ww+a, hh+a])
        else:
            dim_to_pad = 1 if w<h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h,h), color=(127,127,127))
                padded_img.paste(img, (int(padding), 0))
                # lab[:, [1]] = (lab[:, [1]] * w + padding) / h
                # lab[:, [3]] = (lab[:, [3]] * w / h)
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127,127,127))
                padded_img.paste(img, (0, int(padding)))
                # lab[:, [2]] = (lab[:, [2]] * h + padding) / w
                # lab[:, [4]] = (lab[:, [4]] * h  / w)
        resize = transforms.Resize((self.imgsize, self.imgsize))
        padded_img = resize(padded_img)     #choose here
        return padded_img

    def pad_lab(self, lab):
        pad_size = self.max_n_labels - lab.shape[0]
        if(pad_size>0):
            padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=1)
        else:
            padded_lab = lab
        return padded_lab


class PatchTransformer(nn.Module):
    """PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.

    """

    def __init__(self):
        super(PatchTransformer, self).__init__()
        self.min_contrast = 0.9
        self.max_contrast = 1.1
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.02
        self.min_scale = -0.28  # log 0.75
        self.max_scale = 0.47  # log 1.60
        self.translation_x = 0.8
        self.translation_y = 1.0

    def forward(self, img_batch, adv_patch):
        # import matplotlib.pyplot as plt
        B, _, Ht, Wt = img_batch.shape
        _, _, Ho, Wo = adv_patch.shape
        adv_patch = adv_patch[:B]

        mask = (adv_patch[:, -1:, ...] > 0).to(adv_patch)
        adv_patch = adv_patch[:, :-1, ...]

        contrast = adv_patch.new(size=[B]).uniform_(self.min_contrast, self.max_contrast)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        brightness = adv_patch.new(size=[B]).uniform_(self.min_brightness, self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        noise = adv_patch.new(adv_patch.shape).uniform_(-1, 1) * self.noise_factor

        adv_patch = adv_patch * contrast + brightness + noise
        adv_patch = adv_patch.clamp(0, 1)
        adv_patch = torch.cat([adv_patch, mask], dim=1)

        scale = adv_patch.new(size=[B]).uniform_(self.min_scale, self.max_scale).exp()
        mesh_bord = torch.stack([torch.cat([m[0].nonzero().min(0).values, m[0].nonzero().max(0).values]) for m in mask])
        mesh_bord = mesh_bord / mesh_bord.new([Ho, Wo, Ho, Wo]) * 2 - 1
        #         mesh_bord = mesh_bord / scale
        pos_param = mesh_bord + mesh_bord.new([1, 1, -1, -1]) * scale.unsqueeze(-1)
        tymin, txmin, tymax, txmax = pos_param.unbind(-1)

        xdiff = (-txmax + txmin).clamp(min=0)
        xmiddle = (txmax + txmin) / 2
        ydiff = (-tymax + tymin).clamp(min=0)
        ymiddle = (tymax + tymin) / 2

        tx = txmin.new(txmin.shape).uniform_(-0.5, 0.5) * xdiff * self.translation_x + xmiddle
        ty = tymin.new(tymin.shape).uniform_(-0.5, 0.5) * ydiff * self.translation_y + ymiddle

        theta = adv_patch.new_zeros(B, 2, 3)
        theta[:, 0, 0] = scale
        theta[:, 0, 1] = 0
        theta[:, 1, 0] = 0
        theta[:, 1, 1] = scale
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty

        grid = F.affine_grid(theta, img_batch.shape)
        adv_batch = F.grid_sample(adv_patch, grid, padding_mode='zeros')
        mask = adv_batch[:, -1:]
        # print(adv_batch.device, mask.device, img_batch.device)
        adv_batch = adv_batch[:, :-1] * mask + img_batch * (1 - mask)

        gt = torch.stack([torch.cat([m[0].nonzero().min(0).values, m[0].nonzero().max(0).values]) for m in mask])
        gt = gt[:, [1, 0, 3, 2]].unbind(0)
        return adv_batch, gt


class TPSGridGen(nn.Module):
    def __init__(self, target_shape=None, target_control_points=None, target_coordinate=None):
        super(TPSGridGen, self).__init__()
        self.target_shape = target_shape

        assert target_control_points.ndimension() == 2
        self.ndim = target_control_points.size(1)
        N = target_control_points.size(0)
        self.num_points = N
        target_control_points = target_control_points.float()
        self.register_buffer('target_control_points', target_control_points)

        # create padded kernel matrix
        forward_kernel = torch.zeros(N + 1 + self.ndim, N + 1 + self.ndim)
        target_control_partial_repr = self.compute_partial_repr(target_control_points, target_control_points)
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        forward_kernel[:N, N].fill_(1)
        forward_kernel[N, :N].fill_(1)
        forward_kernel[:N, N+1:].copy_(target_control_points)
        forward_kernel[N+1:, :N].copy_(target_control_points.transpose(0, 1))
        
        # compute inverse matrix
        inverse_kernel = torch.inverse(forward_kernel)

        # create target cordinate matrix
        if target_coordinate is None:
            assert self.ndim == 2
            HW = target_shape.numel()
            Y, X = torch.meshgrid(*[torch.linspace(-1, 1, s) for s in target_shape])
            target_coordinate = torch.stack([X.flatten(), Y.flatten()], dim=1) # convert from (y, x) to (x, y)
            target_coordinate_partial_repr = self.compute_partial_repr(target_coordinate, target_control_points)
            target_coordinate_repr = torch.cat([
                target_coordinate_partial_repr, torch.ones(HW, 1), target_coordinate
            ], dim=1)
        else:
            target_coordinate_partial_repr = self.compute_partial_repr(target_coordinate, target_control_points)
            target_coordinate_repr = torch.cat([
                target_coordinate_partial_repr, torch.ones(target_coordinate.shape[0], 1), target_coordinate
            ], dim=1)

        # register precomputed matrices
        self.register_buffer('target_coordinate', target_coordinate.clone().detach())
        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('padding_matrix', torch.zeros(self.ndim + 1, self.ndim))
        self.register_buffer('target_coordinate_repr', target_coordinate_repr)

    def forward(self, source_control_points):
        assert source_control_points.ndimension() == 3
        assert source_control_points.size(1) == self.num_points
        assert source_control_points.size(2) == self.ndim
        batch_size = source_control_points.size(0)

        Y = torch.cat([source_control_points, Variable(self.padding_matrix.expand(batch_size, -1, -1))], 1)
        mapping_matrix = torch.matmul(Variable(self.inverse_kernel), Y)
        new_coordinate = torch.matmul(Variable(self.target_coordinate_repr), mapping_matrix)
        return new_coordinate

    # phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2
    def compute_partial_repr(self, input_points, control_points):
        N = input_points.size(0)
        M = control_points.size(0)
        pairwise_diff = input_points.view(N, 1, self.ndim) - control_points.view(1, M, self.ndim)
        pairwise_dist = (pairwise_diff * pairwise_diff).sum(-1)
        repr_matrix = 0.5 * pairwise_dist * pairwise_dist.log()
        # fix numerical error for 0 * log(0), substitute all nan with 0
        mask = repr_matrix != repr_matrix
        repr_matrix.masked_fill_(mask, 0)
        return repr_matrix

    def tps_mesh(self, source_control_points=None, max_range=(0.1, ), batch_size=1):

        if source_control_points is None:
            source_control_points = self.target_control_points.expand(batch_size, -1, -1)
            source_control_points = source_control_points + source_control_points.new(source_control_points.shape).uniform_(-1, 1) * source_control_points.new(max_range)
            # source_control_points = source_control_points.to(self.padding_matrix.device)
        source_coordinate = self.forward(source_control_points)
        return source_coordinate


    def tps_trans(self, inputs, source_control_points=None, max_range=0.1, canvas=0.5, target_shape=None):
        if target_shape is not None:
            if target_shape != self.target_shape:
                device = self.padding_matrix.device
                self.__init__(target_shape, self.target_control_points.cpu())
                self.to(device)

        target_height, target_width = self.target_shape

        if source_control_points is None:
            source_control_points = self.target_control_points.unsqueeze(0) + self.target_control_points.new(
                size=(inputs.shape[0], ) + self.target_control_points.shape).uniform_(-1, 1) * max_range
            # source_control_points = source_control_points.to(self.padding_matrix.device)
        if isinstance(canvas, float):
            canvas = torch.FloatTensor(inputs.shape[0], inputs.shape[1], target_height, target_width).fill_(canvas).to(self.padding_matrix.device)
        source_coordinate = self.forward(source_control_points)
        grid = source_coordinate.view(inputs.shape[0], target_height, target_width, 2)
        target_image = grid_sample(inputs, grid, canvas)
        return target_image, source_control_points


class CamoSampleGenerator(object):
    def __init__(self, device, batch_size, alpha=10, checkpoints=600, tps2d_range_t=50, tps2d_range_r=0.1, tps3d_range=0.15,
                 num_points_tshirt=60, num_points_trouser=60, save_path='results/yolov3_07', blur=1):
        
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        self.device = device
        self.img_size = 416
        self.DATA_DIR = "./data"

        self.tps2d_range_t = tps2d_range_t
        self.tps2d_range_r = tps2d_range_r
        self.tps3d_range = tps3d_range

        self.batch_size = batch_size

        self.patch_transformer = PatchTransformer().to(device)

        self.alpha = alpha
        self.azim = torch.zeros(self.batch_size)

        self.sampler_probs = torch.ones([36]).to(device)
        color_transform = ColorTransform('color_transform_dim6.npz')
        self.color_transform = color_transform.to(device)

        self.fig_size_H = 340
        self.fig_size_W = 864

        self.fig_size_H_t = 484
        self.fig_size_W_t = 700

        resolution = 4
        h, w, h_t, w_t = int(self.fig_size_H / resolution), int(self.fig_size_W / resolution), int(self.fig_size_H_t / resolution), int(self.fig_size_W_t / resolution)
        self.h, self.w, self.h_t, self.w_t = h, w, h_t, w_t
        num_colors = 4

        # Set paths
        obj_filename_man = os.path.join(self.DATA_DIR, "Archive/Man_join/man.obj")
        obj_filename_tshirt = os.path.join(self.DATA_DIR, "Archive/tshirt_join/tshirt.obj")
        obj_filename_trouser = os.path.join(self.DATA_DIR, "Archive/trouser_join/trouser.obj")

        self.coordinates = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), -1).to(device)
        self.coordinates_t = torch.stack(torch.meshgrid(torch.arange(h_t), torch.arange(w_t)), -1).to(device)
        self.tshirt_point = torch.rand([num_colors, num_points_tshirt, 3], requires_grad=True, device=device)
        self.trouser_point = torch.rand([num_colors, num_points_trouser, 3], requires_grad=True, device=device)
        self.colors = torch.load("data/camouflage4.pth").float().to(device)
        self.mesh_man = load_objs_as_meshes([obj_filename_man], device=device)
        self.mesh_tshirt = load_objs_as_meshes([obj_filename_tshirt], device=device)
        self.mesh_trouser = load_objs_as_meshes([obj_filename_trouser], device=device)

        self.faces = self.mesh_tshirt.textures.faces_uvs_padded()
        self.verts_uv = self.mesh_tshirt.textures.verts_uvs_padded()
        self.faces_uvs_tshirt = self.mesh_tshirt.textures.faces_uvs_list()[0]

        self.faces_trouser = self.mesh_trouser.textures.faces_uvs_padded()
        self.verts_uv_trouser = self.mesh_trouser.textures.verts_uvs_padded()
        self.faces_uvs_trouser = self.mesh_trouser.textures.faces_uvs_list()[0]

        self.seeds_tshirt = torch.zeros(size=[h, w, num_colors], device=device).uniform_()
        self.seeds_trouser = torch.zeros(size=[h_t, w_t, num_colors], device=device).uniform_()

        k = 3
        k2 = k * k
        self.camouflage_kernel = nn.Conv2d(num_colors, num_colors, k, 1, int(k / 2)).to(device)
        self.camouflage_kernel.weight.data.fill_(0)
        self.camouflage_kernel.bias.data.fill_(0)
        for i in range(num_colors):
            self.camouflage_kernel.weight[i, i, :, :].data.fill_(1 / k2)

        self.expand_kernel = nn.ConvTranspose2d(3, 3, resolution, stride=resolution, padding=0).to(device)
        self.expand_kernel.weight.data.fill_(0)
        self.expand_kernel.bias.data.fill_(0)
        for i in range(3):
            self.expand_kernel.weight[i, i, :, :].data.fill_(1)

        selected_tshirt = torch.cat([torch.arange(27), torch.arange(28, 31), torch.arange(32, 43)])
        self.tshirt_locations_infos = EasyDict({
            'nparts': 3,
            'centers': [[7.5, 0], [-7.5, 0], [0, 0]],
            'Rs': [1.5, 1.5, 15.0],
            'ntfs': [6, 6, 8],
            'ntws': [6, 6, 8],
            'radius_fixed': [[1.0], [1.0], [0.5]],
            'radius_wrap': [[0.5], [0.5], [1.0]],
            'signs': [-1, -1, 1],
            'selected': selected_tshirt,
        })

        self.trouser_locations_infos = EasyDict({
            'nparts': 2,
            'centers': [[3.43, 0], [-3.43, 0]],
            'Rs': [3.3] * 2,
            'ntfs': [20] * 2,
            'ntws': [12] * 2,
            'radius_fixed': [[1.2]] * 2,
            'radius_wrap': [[0.4]] * 2,
            'signs': [1, 1],
            'selected': None,
        })

        self.initialize_tps2d()
        self.initialize_tps3d()

        self.load_weights(save_path, checkpoints - 1)
        self.blur = blur
        self.update_mesh(type='determinate')

    def get_loader(self, img_dir, shuffle=True):
        loader = torch.utils.data.DataLoader(InriaDataset(img_dir, self.img_size, shuffle=shuffle),
                                             batch_size=self.batch_size,
                                             shuffle=True,
                                             num_workers=4)
        return loader

    def sample_cameras(self, theta=None, elev=None):
        if theta is not None:
            if isinstance(theta, float) or isinstance(theta, int):
                self.azim = torch.zeros(self.batch_size).fill_(theta)
            elif isinstance(theta, torch.Tensor):
                self.azim = theta.clone()
            elif isinstance(theta, np.ndarray):
                self.azip = torch.from_numpy(theta)
            else:
                raise ValueError
        else:
            if self.alpha > 0:
                exp = (self.alpha * self.sampler_probs).softmax(0)
                azim = torch.multinomial(exp, self.batch_size, replacement=True)
                self.azim_inds = azim
                azim = azim.to(exp)
                self.azim = (azim + azim.new(size=azim.shape).uniform_() - 0.5) * 360 / len(exp)
            else:
                self.azim_inds = None
                self.azim = (torch.zeros(self.batch_size).uniform_() - 0.5) * 360
        if elev is not None:
            elev = torch.zeros(self.batch_size).fill_(elev)
        else:
            elev = 10 + 8 * torch.zeros(self.batch_size).uniform_(-1, 1)
        R, T = look_at_view_transform(dist=2.5, elev=elev, azim=self.azim)
        return FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=45)

    def sample_lights(self, r=None):
        if r is None:
            r = np.random.rand()
        theta = np.random.rand() * 2 * math.pi
        if r < 0.33:
            lights = AmbientLights(device=self.device)
        elif r < 0.67:
            lights = DirectionalLights(device=self.device, direction=[[np.sin(theta), 0.0, np.cos(theta)]])
        else:
            lights = PointLights(device=self.device, location=[[np.sin(theta) * 3, 0.0, np.cos(theta) * 3]])
        return lights

    def initialize_tps2d(self):
        locations_tshirt_ori = torch.load(os.path.join(self.DATA_DIR, 'Archive/tshirt_join/projections/part_all_2p5.pt'), map_location='cpu').to(self.device)
        self.infos_tshirt = MU.get_map_kernel(locations_tshirt_ori, self.faces_uvs_tshirt)

        locations_trouser_ori = torch.load(os.path.join(self.DATA_DIR, 'Archive/trouser_join/projections/part_all_off3p4.pt'), map_location='cpu').to(self.device)
        self.infos_trouser = MU.get_map_kernel(locations_trouser_ori, self.faces_uvs_trouser)

        target_control_points = p3dmd.get_points(self.tshirt_locations_infos, wrap=False).squeeze(0).cpu()
        tps2d_tshirt = TPSGridGen(None, target_control_points, locations_tshirt_ori.cpu())
        tps2d_tshirt.to(self.device)
        self.tps2d_tshirt = tps2d_tshirt

        target_control_points = p3dmd.get_points(self.trouser_locations_infos, wrap=False).squeeze(0).cpu()
        tps2d_trouser = TPSGridGen(None, target_control_points, locations_trouser_ori.cpu())
        tps2d_trouser.to(self.device)
        self.tps2d_trouser = tps2d_trouser
        return

    def initialize_tps3d(self):
        xmin, ymin, zmin = (-0.28170400857925415, -0.7323740124702454, -0.15313300490379333)
        xmax, ymax, zmax = (0.28170400857925415, 0.5564370155334473, 0.0938199982047081)
        xnum, ynum, znum = [5, 8, 5]
        max_range = (torch.Tensor([xmax, ymax, zmax]) - torch.Tensor([xmin, ymin, zmin])) / torch.Tensor(
            [xnum, ynum, znum])
        self.max_range = (max_range * self.tps3d_range).tolist()
        target_control_points = torch.tensor(list(itertools.product(
            torch.linspace(xmin, xmax, xnum),
            torch.linspace(ymin, ymax, ynum),
            torch.linspace(zmin, zmax, znum),
        )))
        mesh = MU.join_meshes([self.mesh_man, self.mesh_tshirt, self.mesh_trouser])

        tps3d = TPSGridGen(None, target_control_points, mesh.verts_packed().cpu())
        tps3d.to(self.device)
        self.tps3d = tps3d
        return

    def synthesis_image(self, img_batch, use_tps2d=True, use_tps3d=True):
        cameras = self.sample_cameras(theta)
        lights = self.sample_lights()
        if use_tps2d:
            # tps_2d
            source_control_points_tshirt = p3dmd.get_points(self.tshirt_locations_infos, torch.pi / 180 * self.tps2d_range_t, self.tps2d_range_r,
                                                            bs=self.batch_size, random=True)
            locations_tshirt = self.tps2d_tshirt(source_control_points_tshirt.to(self.device))
            source_control_points_trouser = p3dmd.get_points(self.trouser_locations_infos, torch.pi / 180 * self.tps2d_range_t, self.tps2d_range_r,
                                                             bs=self.batch_size, random=True)
            locations_trouser = self.tps2d_trouser(source_control_points_trouser.to(self.device))
        else:
            locations_tshirt = locations_trouser = None

        if use_tps3d:
            # tps_3d
            source_coordinate = self.tps3d.tps_mesh(max_range=self.max_range, batch_size=self.batch_size).view(-1, 3)
        else:
            source_coordinate = None
        images_predicted = p3dmd.view_mesh_wrapped([self.mesh_man, self.mesh_tshirt, self.mesh_trouser],
                                                   [None, locations_tshirt, locations_trouser],
                                                   [None, self.infos_tshirt, self.infos_trouser], source_coordinate,
                                                   cameras=cameras, lights=lights, image_size=800, fov=45,
                                                   max_faces_per_bin=30000, faces_per_pixel=3)
        adv_batch = images_predicted.permute(0, 3, 1, 2)
        p_img_batch, gt = self.patch_transformer(img_batch, adv_batch)
        return p_img_batch, gt

    def prob_fix_color(self, original_circles, coordinates, colors, fig_size_h, fig_size_w,blur=1):
        assert original_circles.shape[0] == colors.shape[0]
        coordinates = coordinates.expand(original_circles.shape[1],-1,-1,-1).permute(1,2,0,3)
        circle0 = original_circles[...,0]*fig_size_h
        circle1 = original_circles[...,1]*fig_size_w
        circles = torch.stack([circle0,circle1],dim=-1)
        dist_sum = torch.zeros([colors.shape[0],fig_size_h,fig_size_w]).to(coordinates.device)
        for color_idx in range(colors.shape[0]):
            dist = torch.norm(coordinates-circles[color_idx,:,:2],dim=-1)
            dist_sum[color_idx] = torch.exp(-dist/blur).sum(dim=-1)
        dist_sum = dist_sum/dist_sum.sum(dim=0)
        return dist_sum

    def gumbel_color_fix_seed(self, prob_map, seed, color, tau=0.3, type='gumbel'):
        if type == 'gumbel':
            color_map = F.softmax((torch.log(prob_map) + seed)/tau, dim=-1)
        elif type == 'determinate':
            color_ind = (torch.log(prob_map) + seed).max(-1)[1]
            color_map = F.one_hot(color_ind, prob_map.shape[-1]).to(prob_map)
        else:
            raise ValueError
        tex = torch.matmul(color_map, color).unsqueeze(0)
        return tex

    def update_mesh(self, tau=0.3, type='gumbel'):
        # camouflage:
        prob_map = self.prob_fix_color(self.tshirt_point, self.coordinates, self.colors, self.h, self.w, blur=self.blur).unsqueeze(0)
        prob_trouser = self.prob_fix_color(self.trouser_point, self.coordinates_t, self.colors, self.h_t, self.w_t, blur=self.blur).unsqueeze(0)
        prob_map = self.camouflage_kernel(prob_map)
        prob_trouser = self.camouflage_kernel(prob_trouser)
        prob_map = prob_map.squeeze(0).permute(1, 2, 0)
        prob_trouser = prob_trouser.squeeze(0).permute(1, 2, 0)

        gb_tshirt = -(-(self.seeds_tshirt + 1e-20).log() + 1e-20).log()
        gb_trouser = -(-(self.seeds_trouser + 1e-20).log() + 1e-20).log()

        tex = self.gumbel_color_fix_seed(prob_map, gb_tshirt, self.colors, tau=tau, type=type)
        tex_trouser = self.gumbel_color_fix_seed(prob_trouser, gb_trouser, self.colors, tau=tau, type=type)

        tex = self.expand_kernel(self.color_transform(tex.permute(0, 3, 1, 2))).permute(0, 2, 3, 1)
        tex_trouser = self.expand_kernel(self.color_transform(tex_trouser.permute(0, 3, 1, 2))).permute(0, 2, 3, 1)

        self.mesh_tshirt.textures = TexturesUV(maps=tex, faces_uvs=self.faces, verts_uvs=self.verts_uv)
        self.mesh_trouser.textures = TexturesUV(maps=tex_trouser, faces_uvs=self.faces_trouser, verts_uvs=self.verts_uv_trouser)

        return tex, tex_trouser

    def load_weights(self, save_path, epoch):
        path = save_path + '/' + str(epoch) + '_circle_epoch.pth'
        self.tshirt_point.data = torch.load(path, map_location='cpu').to(self.device)

        path = save_path + '/' + str(epoch) + '_color_epoch.pth'
        self.colors.data = torch.load(path, map_location='cpu').to(self.device)

        path = save_path + '/' + str(epoch) + '_trouser_epoch.pth'
        self.trouser_point.data = torch.load(path, map_location='cpu').to(self.device)

        path = save_path + '/' + str(epoch) + '_seed_tshirt_epoch.pth'
        self.seeds_tshirt = torch.load(path, map_location='cpu').to(self.device)

        path = save_path + '/' + str(epoch) + '_seed_trouser_epoch.pth'
        self.seeds_trouser = torch.load(path, map_location='cpu').to(self.device)


if __name__ == '__main__':
    device, batch_size = 'cuda:0', 4

    generator = CamoSampleGenerator(device, batch_size)
    trainloader = generator.get_loader('./data/background_test', True)
    
    data = next(iter(trainloader))
    data = data.to(device)

    theta = 45

    p_img_batch, gt = generator.synthesis_image(data)
    transforms.ToPILImage()(p_img_batch[0]).save('tbd.png')

