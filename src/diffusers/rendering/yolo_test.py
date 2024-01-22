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
    device = 'cuda:7'
    adv_model = 'yolov2'
    img_size = 416
    pipeline = '3d'
    
    trainloader, _ = get_dataloader(adv_batch_size, pipeline=pipeline)
    yolo = get_model(None, device, adv_model)  # Yolo
    renderer = get_renderer(device, True)

    data, _ = next(iter(trainloader))
    # for batch_idx, (data, _) in enumerate(trainloader):
    data = data.to(device, non_blocking=True)
    img_paths = ['src/diffusers/basic.png']#, 'patches/basic_75_grey_lady.png']
    imgs = [PILToTensor()(Image.open(i)).unsqueeze(0) / 256 for i in img_paths]
    adv_patch = imgs[0].to(device)
    adv_imgs, targets_padded = get_adv_imgs(adv_patch, pipeline, None, None, None,
                                            None, data, renderer, 0,
                                            trainloader)
    colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]])
    fontc = (255, 255, 255)

    ind = 0
    boxes = targets_padded[ind].clone()
    boxes = torch.nn.functional.pad(boxes, (0, 2)).roll(6, 1)

    import math
    from PIL import Image, ImageDraw, ImageFont
    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)


    img = ToPILImage()(adv_imgs[ind])
    width = img.width
    height = img.height
    draw = ImageDraw.Draw(img)
    for i in range(len(boxes)):
        box = boxes[i]
        # box = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]]
        x1 = (box[0] - box[2] / 2.0) * width
        y1 = (box[1] - box[3] / 2.0) * height
        x2 = (box[0] + box[2] / 2.0) * width
        y2 = (box[1] + box[3] / 2.0) * height

        rgb = (255, 0, 0)
        cls_id = int(boxes[i][6])
        if cls_id == 0:
            draw.rectangle([x1, y1, x2, y2], outline=rgb)
            # draw.rectangle([boxw[0], boxw[1], boxw[2], boxw[3]], outline=(0, 255, 0))
    img.save('tbd.png')
