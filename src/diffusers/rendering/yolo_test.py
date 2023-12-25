from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch import Tensor
import torch
import os.path as osp
from torchvision.transforms import ToPILImage, PILToTensor
from PIL import Image
from torch.cuda import amp

from src.diffusers.adversarial.load_target_model import get_dataloader, get_model
from src.diffusers.adversarial.load_target_model import get_renderer, get_dataloader, get_adv_imgs

if __name__ == '__main__':
    adv_batch_size = 24
    device = 'cuda:9'
    adv_model = 'yolov2'
    img_size = 416
    pipeline = '3d'
    
    trainloader, _ = get_dataloader(adv_batch_size, pipeline=pipeline)
    yolo = get_model(None, device, adv_model, pipeline)  # Yolo
    renderer, textures = get_renderer(device)

    camou_tshirt, camou_trouser = textures
    data, _ = next(iter(trainloader))
    # for batch_idx, (data, _) in enumerate(trainloader):
    data = data.to(device, non_blocking=True)
    img_paths = ['patches/basic_75_young_woman.png', 'patches/basic_75_grey_lady.png']
    imgs = [PILToTensor()(Image.open(i)).unsqueeze(0) / 256 for i in img_paths]
    adv_patch = imgs[0].to(device)
    adv_imgs, targets_padded = get_adv_imgs(adv_patch, pipeline, None, None, None,
                                            None, data, renderer, 0,
                                            trainloader, textures)
    ToPILImage()(adv_imgs[0]).save('tbd.png')
