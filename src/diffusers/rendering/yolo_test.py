from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch import Tensor
import torch
import os.path as osp
from torchvision.transforms import ToPILImage, PILToTensor
from PIL import Image, ImageDraw
from torch.cuda import amp

from src.diffusers.adversarial.load_target_model import get_dataloader, get_model
from src.diffusers.adversarial.load_target_model import get_renderer, get_dataloader, get_adv_imgs

from yolo2.utils import get_region_boxes_general


def back_text(draw, x, y, msg, backc, fontc, font=None):
    if font is None:
        font = draw.getfont()
    _, _, w, h = font.getbbox(msg)
    draw.rectangle((x, y, x+w, y+h), fill=backc)
    draw.text((x, y), msg, fill=fontc)
    return None

class Gradient:
    def __init__(self, op, ed, n):
        self.op = op 
        self.ed = ed
        self.n = n
        self.i = 0

    def __call__(self):
        ratio = self.i / self.n 
        self.i += 1
        return tuple([r.item() for r in (self.op * ratio + self.ed * (1 - ratio)).int()])


def get_draw_boxes(yolo, adv_img, draw, conf_thresh, name, grad0, grad1, fontc):
    print('yolo running...')
    output = yolo(adv_img.unsqueeze(0))
    print('yolo finished')
    
    width = ToPILImage()(adv_img).width
    height = ToPILImage()(adv_img).height

    all_boxes = get_region_boxes_general(output, yolo, conf_thresh=conf_thresh, name=name)

    boxes = all_boxes[0] 
    boxes = boxes[boxes[:, 4].sort(axis=0).indices]

    gradient = Gradient(grad0, grad1, len(boxes))
    print(boxes[boxes[:, 6] == 0])
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = (box[0] - box[2] / 2.0) * width
        y1 = (box[1] - box[3] / 2.0) * height
        x2 = (box[0] + box[2] / 2.0) * width
        y2 = (box[1] + box[3] / 2.0) * height

        rgb = gradient()
        cls_id = box[6]
        if cls_id == 0:
            draw.rectangle([x1, y1, x2, y2], outline=rgb)
            back_text(draw, x1, y1, "%.3f" % (box[4]), backc=rgb, fontc=fontc)

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

if __name__ == '__main__':
    adv_batch_size = 24
    device = 'cuda:7'
    adv_model = 'yolov3'
    img_size = 416
    pipeline = '3d'
    conf_thresh = 0.3
    
    print('getting dataloader & renderer...')
    trainloader, _ = get_dataloader(adv_batch_size, pipeline=pipeline)
    renderer = get_renderer(device, True)
    print('got dataloader & renderer')

    data, _ = next(iter(trainloader))
    # for batch_idx, (data, _) in enumerate(trainloader):
    data = data.to(device, non_blocking=True)
    img_paths = ['patches/space_clothes_pattern/adv_256_7_0:4000_225:2000_3d_space_clothes_pattern.png'] #, 'patches/basic_75_grey_lady.png']
    img_paths = ['patches/space_clothes_pattern/adv_256_7_0:10000_3d_yolov3_space_clothes_pattern.png'] #, 'patches/basic_75_grey_lady.png']
    # img_paths = ['process/29.png'] #, 'patches/basic_75_grey_lady.png']
    imgs = [PILToTensor()(Image.open(i)).unsqueeze(0) / 256 for i in img_paths]
    adv_patch = imgs[0].to(device)
    print('Getting adv images...')
    adv_imgs, targets_padded = get_adv_imgs(adv_patch, pipeline, None, None, None,
                                            None, data, renderer, 0,
                                            trainloader)
    print('Got adv images')
    colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]])
    fontc = (255, 255, 255)

    ind = 0
    boxes = targets_padded[ind].clone()
    boxes = torch.nn.functional.pad(boxes, (0, 2)).roll(6, 1)
    print(boxes.shape)
    boxes[..., 4] = 1

    img = ToPILImage()(adv_imgs[ind])
    width = img.width
    height = img.height
    draw = ImageDraw.Draw(img)
    
    img1 = ToPILImage()(adv_imgs[ind])
    draw1 = ImageDraw.Draw(img1)

    for i in range(len(boxes)):
        box = boxes[i]
        x1 = (box[0] - box[2] / 2.0) * width
        y1 = (box[1] - box[3] / 2.0) * height
        x2 = (box[0] + box[2] / 2.0) * width
        y2 = (box[1] + box[3] / 2.0) * height

        rgb = (255, 0, 0)
        cls_id = int(boxes[i][6])
        if cls_id == 0:
            draw.rectangle([x1, y1, x2, y2], outline=rgb)
            draw1.rectangle([x1, y1, x2, y2], outline=rgb)
            
    yolo2 = get_model(None, torch.device(device), 'yolov2')  # Yolo
    get_draw_boxes(yolo2, adv_imgs[0], draw, conf_thresh, 'yolov2', torch.Tensor([255, 255, 255]), torch.Tensor([0, 0, 255]), (0, 0, 0))
    
    yolo3 = get_model(None, torch.device(device), 'yolov3')  # Yolo
    get_draw_boxes(yolo3, adv_imgs[0], draw1, conf_thresh, 'yolov3', torch.Tensor([00, 0, 0]), torch.Tensor([0, 200, 0]), (255, 255, 255))
    # yolov3: [(bs, nboxes, 85=xywh_conf_classes), [maps x3]]

    get_concat_h(img, img1).save('tbd.png')
