import torch
import yaml
import warnings

from utils.google_utils import attempt_download
from yolov7.utils.general import check_file, check_dataset
from yolov7.utils.torch_utils import intersect_dicts
from yolov7.data import load_data
from yolov7.models.yolo import Model
from yolo2.darknet import Darknet

import sys

# Load Inria dataset
def get_dataloader(adv_batch_size, adv_model='yolov2'):
    if adv_model != 'yolov2':
        warnings.warn(f"The dataset is customized for yolov2, but used {adv_model}")
    imgsz = 416  # standard for Inria
    with open(check_file('data/inria.yaml')) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    check_dataset(data_dict, True)
    adv_dataset = load_data.InriaDataset(data_dict['train_data'], data_dict['train_labels'], imgsz, shuffle=True)
    adv_dataloader = torch.utils.data.DataLoader(adv_dataset, batch_size=adv_batch_size, shuffle=True, num_workers=8,
                                                    collate_fn=load_data.collate_wo_max_label)
    return adv_dataloader, data_dict


def get_model(data_dict, device, adv_model='yolov2'):
    if adv_model == 'yolov7':
        sys.path.insert(0, './yolov7')  # to get the model weights
        weights = 'yolov7/yolov7.pt'
        attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        hyp = check_file('data/hyp.scratch.p5.yaml')
        with open(hyp) as f:  # Hyperparameters
            hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
        nc = int(data_dict['nc'])  # number of classes
        yolo = Model(ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, yolo.state_dict())
        yolo.load_state_dict(state_dict, strict=False)  # load
        yolo.hyp = hyp
        yolo.hyp['box'], yolo.hyp['cls'] = 0, 0  # boxes don't matter, neither does (mis)classification
        yolo.hyp['obj'] = 1  # we just want to avoid detection
        yolo.nc = nc  # attach number of classes to model
        yolo.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
        yolo.names = data_dict['names']
    elif adv_model == 'yolov2':
        # sys.path.insert(0, './yolo2')  # to get the model weights
        cfgfile = 'yolo2/yolov2.cfg'
        weights = 'yolo2/yolov2.weights'
        attempt_download(weights, 'https://pjreddie.com/media/files/yolov2.weights', True)
        yolo = Darknet(cfgfile)
        yolo.load_weights(weights)
        yolo = yolo.to(device)
    else:
        raise ValueError(f"No model named {adv_model}")

    for _, v in yolo.named_parameters():  # freeze all layers
        v.requires_grad = False
    yolo.eval()
    return yolo
