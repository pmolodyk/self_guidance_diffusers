import torch
import yaml
import warnings
from os import path
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision.models import detection

from src.diffusers.adversarial.utils.google_utils import attempt_download
from yolo2.darknet import Darknet
from yolo2.utils import get_det_loss
from yolo3.yolov3_models import YOLOv3Darknet
from yolov7.utils.general import check_file, check_dataset
from yolov7.utils.loss import ComputeLoss as ComputeLossv7
from yolov7.data import load_data

from src.diffusers.adversarial.utils.yolo_dataset_utils import targets2padded
from src.diffusers.rendering import RenderState
from src.diffusers.rendering.camera import CameraSampler
from src.diffusers.rendering.person_model import PersonModel
from src.diffusers.rendering.texture.i_texture import ITexture
from src.diffusers.rendering.texture.simple_texture import SimpleTexture
from src.diffusers.rendering.image_synthesizer import ImageSynthesizer
from src.diffusers.rendering.light import LightSampler


# Load Inria dataset
def get_dataloader(adv_batch_size, adv_model='yolov2', pipeline='3d'):
    if adv_model != 'yolov2':
        warnings.warn(f"The dataset is customized for yolov2, but used {adv_model}")
    img_size = 416  # standard for Inria
    if pipeline == 'standard':
        with open(check_file('data/inria.yaml')) as f:
            data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
        check_dataset(data_dict, True)
        adv_dataset = load_data.InriaDataset(data_dict['train_data'], data_dict['train_labels'], img_size, shuffle=True)
        adv_dataloader = torch.utils.data.DataLoader(adv_dataset, batch_size=adv_batch_size, shuffle=True, num_workers=8,
                                                        collate_fn=load_data.collate_wo_max_label)
    elif pipeline == '3d':
        transform = T.Compose(
            [
                T.Resize(img_size),
                T.CenterCrop((img_size, img_size)),
                T.ToTensor(),
            ]
        )
        trainset = ImageFolder(
            path.join("data", "background"), transform=transform
        )
        adv_dataloader = torch.utils.data.DataLoader(
            trainset,
            adv_batch_size,
            True,
            num_workers=10,
            pin_memory=True,
        )
        data_dict = None
    else:
        raise ValueError(f"Unknown pipeline {pipeline}")
    return adv_dataloader, data_dict


def get_model(data_dict, device, adv_model='yolov2'):
    if adv_model == 'yolov3':
        yolo = YOLOv3Darknet().to(device)
        weights = 'data/yolov3.weights'
        attempt_download(weights, 'https://pjreddie.com/media/files/yolov3.weights', True)
        yolo.load_darknet_weights(weights)
    elif adv_model == 'yolov2':
        cfgfile = 'yolo2/yolov2.cfg'
        weights = 'yolo2/yolov2.weights'
        attempt_download(weights, 'https://pjreddie.com/media/files/yolov2.weights', True)
        yolo = Darknet(cfgfile)
        yolo.load_weights(weights)
        yolo = yolo.to(device)
    elif adv_model == 'faster-rcnn':
        yolo = detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    elif adv_model == 'detr':
        yolo = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', 
                              pretrained=True).to(device)
    else:
        raise ValueError(f"No model named {adv_model}")

    for _, v in yolo.named_parameters():  # freeze all layers
        v.requires_grad = False
    yolo.eval()
    return yolo


def get_adv_imgs(adv_patch, pipeline, targets_all, tps=None, patch_transformer=None, 
                 patch_applier=None, imgs=None, renderer=None, batch_idx=0, trainloader=None):
    device = adv_patch.device
    imgsz = 416
    if pipeline == '3d':
        adv_imgs, targets = renderer.forward(
            imgs,
            resample=batch_idx % 20 == 0 or batch_idx == len(trainloader) - 1,
            share_texture=True,
            tex_kwargs=dict(tex_map=adv_patch),
            render_kwargs=dict(use_tps2d=True, use_tps3d=True),
        )
        targets_padded = targets2padded(targets)
    elif pipeline == 'standard':
        targets, targets_padded = targets_all
        adv_patch_tps, _ = tps.tps_trans(adv_patch, max_range=0.1, canvas=0.5, target_shape=adv_patch.shape[-2:])
        adv_batch_t = patch_transformer(adv_patch_tps, targets_padded.to(device), imgsz, do_rotate=True, rand_loc=False,
                        pooling='median', old_fasion=False)
        adv_imgs = patch_applier(imgs.float(), adv_batch_t)
    return adv_imgs, targets_padded


def get_renderer(device, eval):
    # Construct model
    person = PersonModel("data", device=device)
    patch_crops: list[torch.nn.Module] = []
    textures: list[ITexture] = []
    for i, cloth in enumerate(person.clothes):
        map_size = cloth.fig_size
        patch_crop = torch.nn.Identity()
        patch_crop_test = patch_crop
        patch_crops.extend([patch_crop, patch_crop_test])
        texture = SimpleTexture(map_size, map_size=torch.Size([256, 256]), tile=True, tile_size=128, eval=eval)
        textures.append(texture.to(device))

    img_synthesizer = ImageSynthesizer(
        contrast=(0.9, 1.1),
        brightness=(-0.1, 0.1),
        noise_factor=0.02,
        scale=(0.75, 1.6),
        translation=(0.8, 1),
        pooling="gauss",
    ).to(device)

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
    return renderer


def get_loss_fn(adv_model, yolo):
    if adv_model in ('yolov2', 'yolov3', 'faster-rcnn'):
        return get_det_loss
    elif adv_model == 'yolov7':
        return ComputeLossv7(yolo)
    else:
        raise ValueError(f"invalid adv_model {adv_model}")
