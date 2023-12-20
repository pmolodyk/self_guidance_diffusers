from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch import Tensor
import torch
import os.path as osp

from src.diffusers.adversarial.load_target_model import get_dataloader, get_model
from yolov7.data import load_data
from yolov7.utils.loss import ComputeLoss
from yolo2.utils import get_det_loss
from .person_model import PersonModel
from .texture.i_texture import ITexture
from .texture.simple_texture import SimpleTexture
from ..models.nps_score import NonPrintablityScore
from .image_synthesizer import ImageSynthesizer
from . import RenderState
from .light import LightSampler
from .camera import CameraSampler

if __name__ == '__main__':
    adv_batch_size = 2
    device = 'cuda:9'
    adv_model = 'yolov2'
    img_size = 416

    # load data
    adv_dataloader, data_dict = get_dataloader(adv_batch_size)  # Inria
    yolo = get_model(data_dict, device, adv_model)  # Yolo
    compute_loss = ComputeLoss(yolo) if adv_model == 'yolov7' else get_det_loss
    patch_transformer = load_data.PatchTransformer().to(device)
    patch_applier = load_data.PatchApplier().to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )
    testset = ImageFolder(
        osp.join("data", "background_test"), transform=transform
    )
    testloader = DataLoader(
        testset,
        adv_batch_size,
        False,
        num_workers=10,
        pin_memory=True,
    )

    # Construct model
    person = PersonModel("data", device=device)
    patch_crops: list[torch.nn.Module] = []
    textures: list[ITexture] = []
    for cloth in person.clothes:
        map_size = cloth.fig_size
        patch_crop = torch.nn.Identity()
        patch_crop_test = patch_crop
        patch_crops.extend([patch_crop, patch_crop_test])
        texture = SimpleTexture(map_size, None, torch.nn.Identity())
        torch.nn.init.uniform_(texture.tex_map)
        textures.append(texture.to(device))

    img_synthesizer = ImageSynthesizer(
        contrast=(0.9, 1.1),
        brightness=(-0.1, 0.1),
        noise_factor=0.02,
        scale=(0.75, 1.6),
        translation=(0.8, 1),
        pooling="gauss",
    ).to(device)
    # img_synthesizer_test = img_synthesizer

    camera_sampler = CameraSampler(device=device)
    light_sampler = LightSampler(device=device)
    renderer = RenderState(
        person,
        camera_sampler,
        light_sampler,
        textures,
        patch_crops,
        img_synthesizer,
    )

    nps_calculator = NonPrintablityScore("data/nps30").to(device)
    trainset = ImageFolder(
            osp.join("data", "background"), transform=transform
        )
    trainloader = DataLoader(
            trainset,
            adv_batch_size,
            True,
            num_workers=10,
            pin_memory=True,
        )
    camou_tshirt, camou_trouser = textures
    for batch_idx, (data, _) in enumerate(trainloader):
        data = data.to(device)
        render_kwargs = dict(use_tps2d=True, use_tps3d=True)
        data, targets = renderer.forward(
            data,
            resample=batch_idx % 20 == 0 or batch_idx == len(trainloader) - 1,
            share_texture=False,
            tex_kwargs=dict(),
            render_kwargs=render_kwargs,
        )
        print(data.shape)
        from torchvision.transforms import ToPILImage
        ToPILImage()(data[0]).save('tbd.png')
        break
