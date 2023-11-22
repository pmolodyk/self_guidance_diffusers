import argparse
import math
import os
import random
import torch
import torch.nn.functional as F
import yaml

from torch.cuda import amp
from tqdm import tqdm
from yolov7.data import load_data
from yolov7.models.yolo import Model
from yolov7.utils.general import init_seeds, check_dataset, check_file
from diffusers.adversarial.utils.google_utils import attempt_download
from yolov7.utils.loss import ComputeLoss
from yolov7.utils.torch_utils import select_device, intersect_dicts


def train(opt, hyp, device):
    # Configure
    cuda = device.type != 'cpu'
    init_seeds(3407)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict

    nc = int(data_dict['nc'])  # number of classes
    names = data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Model
    weights = opt.weights
    attempt_download(weights)  # download if not found locally
    ckpt = torch.load(weights, map_location=device)  # load checkpoint
    model = Model(ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    state_dict = ckpt['model'].float().state_dict()  # to FP32
    state_dict = intersect_dicts(state_dict, model.state_dict())  # intersect
    model.load_state_dict(state_dict, strict=False)  # load

    check_dataset(data_dict, True)  # check

    # Freeze
    for k, v in model.named_parameters():
        v.requires_grad = False  # freeze all layers

    # Image sizes
    imgsz = opt.img_size[0]
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)

    # Trainloader
    batch_size = opt.batch_size
    dataset = load_data.InriaDataset(data_dict['train_data'], data_dict['train_labels'], imgsz,
                                     shuffle=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=opt.workers,
                                             collate_fn=load_data.collate_wo_max_label)

    # Model parameters
    model.hyp = hyp
    model.hyp['box'] *= 0  # boxes don't matter
    model.hyp['cls'] *= 0  # (mis)classification doesn't matter
    model.hyp['obj'] = 1  # we just want to avoid detection
    model.hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.names = names

    # Start training
    compute_loss = ComputeLoss(model)  # init loss class
    print(f'Image sizes {imgsz} train, {opt.img_size[1]} test\n'
          f'Using {dataloader.num_workers} dataloader workers\n')

    model.train()

    pbar = enumerate(dataloader)
    nb = len(dataloader)
    pbar = tqdm(pbar, total=nb)  # progress bar
    for i, (imgs, targets) in pbar:  # batch -------------------------------------------------------------
        imgs = imgs.to(device, non_blocking=True).float()

        # Multi-scale
        if opt.multi_scale:
            sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

        # Forward
        with amp.autocast(enabled=cuda):
            pred = model(imgs)  # [(bs, c, _ * 4, _ * 4, nc + 5), (bs, _ * 2, _ * 2, nc + 5), (bs, _, _, nc + 5)]
            loss, _ = compute_loss(pred, targets.to(device))  # loss scaled by batch_size

        # Backward
        loss.backward()
        1/0  # this code does nothing, I only wanted to make YOLO work on Inria

        # end batch ------------------------------------------------------------------------------------------------
    # end epoch ----------------------------------------------------------------------------------------------------

    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov7.pt', help='weights path')
    parser.add_argument('--data', type=str, default='data/inria.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.p5.yaml', help='hyperparameters path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[416, 416], help='[train, test] image sizes')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')  # todo not sure it works
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    opt = parser.parse_args()

    # Set DDP variables
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    opt.data, opt.hyp = check_file(opt.data), check_file(opt.hyp)

    device = select_device(opt.device, batch_size=opt.batch_size)

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
        if 'anchors' not in hyp:  # anchors commented in hyp.yaml
            hyp['anchors'] = 3

    # run 'train'
    train(opt, hyp, device)
