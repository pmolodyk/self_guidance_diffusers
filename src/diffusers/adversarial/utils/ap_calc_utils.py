import argparse
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import yaml

from decimal import Decimal
from PIL import Image
from os import path
from scipy.interpolate import interp1d
from src.diffusers.adversarial.load_target_model import get_renderer
from src.diffusers.adversarial.utils.yolo_dataset_utils import targets2padded
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from tqdm import tqdm
from torchvision import transforms
from yolo2 import utils, load_data
from yolov7.utils.general import check_file
from ..load_target_model import get_model


def truths_length(truths):
    for i in range(50):
        if truths[i][1] == -1:
            return i


def label_filter(truths, labels=None):
    if labels is not None:
        new_truths = truths.new(truths.shape).fill_(-1)
        c = 0
        for t in truths:
            if t[0].item() in labels:
                new_truths[c] = t
                c = c + 1
        return new_truths


def test(model, loader, adv_patch=None, conf_thresh=0.5, nms_thresh=0.4, iou_thresh=0.5, num_of_samples=100,
         old_fasion=True, pipeline='3d', num_samples=18, device='cuda:0', net='yolov2', img_size=416):
    model.eval()
    total = 0.0
    rend = pipeline == '3d'
    theta_list = [None]
    device = torch.device(device)
    patch_applier = load_data.PatchApplier().to(device)
    patch_transformer = load_data.PatchTransformer().to(device)
    noise = torch.rand(1, 3, 256, 256).to(device)
    if rend:  # render images
        renderer = get_renderer(device, False)
        theta_list = np.linspace(-180, 180, num_samples, endpoint=False)
        renderer.lights = renderer.light_sampler.sample(0)
    with torch.no_grad():
        positives = []
        for batch_idx, (data, target) in tqdm(enumerate(loader), total=len(loader)):
            data = data.to(device)
            if adv_patch is not None:
                target = target.to(device)
                if not rend:
                    adv_batch_t = patch_transformer(adv_patch, target, img_size, do_rotate=True, rand_loc=False,
                                                    pooling='median', old_fasion=old_fasion)
                    data = patch_applier(data, adv_batch_t)
            for angle_idx, theta in enumerate(theta_list):
                if rend:
                    renderer.cameras = renderer.camera_sampler.sample(len(data), theta=theta)
                    if adv_patch is not None:
                        tex_kwargs = dict(tex_map=adv_patch.unsqueeze(0))  # ?
                    else:
                        tex_kwargs = dict(tex_map=noise)
                    render_kwargs = dict(use_tps2d=True, use_tps3d=True)
                    data_render, target = renderer.forward(
                        data,
                        resample=False,
                        is_test=True,
                        share_texture=True,
                        tex_kwargs=tex_kwargs,
                        render_kwargs=render_kwargs,
                        )
                    target = targets2padded(target)
                else:
                    data_render = data
                output = model(data_render)
                all_boxes = utils.get_region_boxes_general(output, model, conf_thresh, net)
                for i in range(len(all_boxes)):
                    boxes = all_boxes[i]
                    boxes = utils.nms(boxes, nms_thresh)

                    truths = target[i].view(-1, 5)
                    truths = label_filter(truths, labels=[0])
                    num_gts = truths_length(truths)
                    truths = truths[:num_gts, 1:]
                    truths = truths.tolist()
                    total = total + num_gts
                    for j in range(len(boxes)):
                        if boxes[j][6].item() == 0:
                            best_iou = 0
                            best_index = 0

                            for ib, box_gt in enumerate(truths):
                                iou = utils.bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                                if iou > best_iou:
                                    best_iou = iou
                                    best_index = ib
                            if best_iou > iou_thresh:
                                del truths[best_index]
                                positives.append((boxes[j][4].item(), True))
                            else:
                                positives.append((boxes[j][4].item(), False))

        positives = sorted(positives, key=lambda d: d[0], reverse=True)
        tps = []
        fps = []
        confs = []
        tp_counter = 0
        fp_counter = 0
        for pos in positives:
            if pos[1]:
                tp_counter += 1
            else:
                fp_counter += 1
            tps.append(tp_counter)
            fps.append(fp_counter)
            confs.append(pos[0])

        precision = []
        recall = []
        for tp, fp in zip(tps, fps):
            recall.append(tp / total)
            precision.append(tp / (fp + tp))

    if len(precision) > 1 and len(recall) > 1:
        p = np.array(precision)
        r = np.array(recall)
        p_start = p[np.argmin(r)]
        samples = np.arange(0., 1., 1.0 / num_of_samples)
        interpolated = interp1d(r, p, fill_value=(p_start, 0.), bounds_error=False)(samples)
        avg = sum(interpolated) / len(interpolated)
    elif len(precision) > 0 and len(recall) > 0:
        avg = precision[0] * recall[0]
    else:
        avg = float('nan')

    return precision, recall, avg, confs


def get_save_aps(device, load_path=None, mask=None, net='yolov2', batch_size=64, no_save_res=False):
    assert net in ("yolov2", "yolov3", "faster-rcnn")
    with open(check_file('data/inria.yaml')) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    darknet_model = get_model(data_dict, torch.device(device), net)

    img_size = 416  # standard for Inria
    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )
    test_data = ImageFolder(
        path.join("data", "background_test"), transform=transform
    )
    loader = torch.utils.data.DataLoader(
        test_data,
        batch_size,
        shuffle=False,
        num_workers=10,
        pin_memory=True,
    )

    res = []
    if load_path is not None:
        if os.path.isdir(load_path):
            if mask is None:
                img_paths = glob.glob(f'{load_path}/*.png')
            else:
                img_paths = glob.glob(f'{load_path}/{mask}')
            print(img_paths)
        else:
            img_paths = [load_path]

        for i, img_path in tqdm(enumerate(img_paths), total=len(img_paths)):
            split_path = img_path.split('/')
            path_to_yaml = '/'.join(split_path[:-1])
            patch_name = split_path[-1]
            aps_name = "aps" if net == "yolov2" else f"aps_{net}"
            
            if not os.path.isfile(f'{path_to_yaml}/{aps_name}.yaml'):
                with open(f'{path_to_yaml}/{aps_name}.yaml', 'w') as f:
                    f.write('aps:\n')
                    f.write('    none: 0')
            with open(f'{path_to_yaml}/{aps_name}.yaml', 'r') as f:
                calculated = yaml.load(f, Loader=yaml.SafeLoader)
            if patch_name in calculated['aps']:
                res.append(calculated['aps'][patch_name])
                continue

            try:
                patch = torch.from_numpy(np.load(img_path)[:1]).to(device)
            except ValueError:
                patch = Image.open(img_path)
                patch = transforms.ToTensor()(patch).to(device)

            test_patch = patch.detach().clone()
            prec, rec, ap, confs = test(darknet_model, loader, adv_patch=test_patch, conf_thresh=0.01, old_fasion=True,
                                        pipeline='3d', device=device, net=net)
            res.append(ap)
            if not no_save_res:
                with open(f'{path_to_yaml}/{aps_name}.yaml', 'a') as f:
                    f.write('\n')
                    f.write(f"    {patch_name}: " + '%.5f'% ap)
    return res


def get_with_mask(load_path, mask=r".+", met_cnt=True, device='cuda:0', calc_ap=True, model_name=''):
    assert model_name in ("yolov2", "yolov3", "faster-rcnn")
    aps_name = "aps" if model_name == "yolov2" else f"aps_{model_name}"
    with open(f'{load_path}/{aps_name}.yaml', 'r') as f:
        calculated = yaml.load(f, Loader=yaml.SafeLoader)
    
    img_paths = glob.glob(f'{load_path}/{mask}')
    print('Checking whether the APs are precalculated...')
    for img_path in tqdm(img_paths):
        patch_name = img_path.split('/')[-1]
        if patch_name not in calculated['aps'] and calc_ap:
            get_save_aps(device, load_path, patch_name)
 
    with open(f'{load_path}/{aps_name}.yaml', 'r') as f:
        calculated = yaml.load(f, Loader=yaml.SafeLoader)
    to_plot = []

    for img_path in img_paths:
        patch_name = img_path.split('/')[-1]
        ap = calculated['aps'][patch_name] if patch_name in calculated['aps'] else -1
        path_split = patch_name.split('_') 
        n = int(path_split[1])
        path_to = '/'.join(img_path.split('/')[:-1])
        original_image = path_to + '/basic_' + str(n) + img_path.split('3d')[-1].replace("_yolov3", "").replace("_faster-rcnn", "")
        l2l = 0
        if met_cnt:
            if 'basic' in img_path:
                l2l = 1e9
            else:
                l2l = -((ToTensor()(Image.open(img_path)).flatten() - 
                         ToTensor()(Image.open(original_image)).flatten()) ** 2).sum().sqrt()
        dct = path_split[3:]
        dct = dct[:sum([':' in el for el in dct])]
        steps = np.array([int(el.split(':')[0]) for el in dct] + [n])
        step_diff = steps[1:] - steps[:-1]
        values = np.array([int(el.split(':')[1]) for el in dct])
        ys = np.repeat(values, step_diff)
        to_plot.append((ap, ys, img_path, n, l2l))
    return to_plot


def get_sg_coef(x):
    patch_name = x[2].split('/')[-1].split('_')
    if 'fx' not in patch_name and 'fa' not in patch_name:
        return 0
    if 'fx' in patch_name:
        return float(patch_name[patch_name.index('fx') + 1])
    return float(patch_name[patch_name.index('fa') - 1])
    
def get_fa_coef(x):
    patch_name = x[2].split('/')[-1].split('_')
    if 'fa' not in patch_name:
        return 0
    return float(patch_name[patch_name.index('fa') + 1]) * get_sg_coef(x)

def get_lo_num(x):
    patch_name = x[2].split('/')[-1].split('_')
    if 'lo' not in patch_name:
        return 0
    return int(patch_name[patch_name.index('lo') + 1])

def get_lo_coef(x):
    patch_name = x[2].split('/')[-1].split('_')
    if 'lo' not in patch_name:
        return 0
    return float(patch_name[patch_name.index('lo') + 2])

def get_yolo_version(x):
    patch_name = x[2].split('/')[-1].split('_')
    if 'yolov3' in patch_name:
        return 3
    return 2

def get_scheduler(x):
    patch_name = x[2].split('/')[-1].split('_')
    sched = " "
    ind = 3
    while ':' in patch_name[patch_name.index('adv') + ind]:
        sched += patch_name[patch_name.index('adv') + ind] + '_'
        ind += 1
    return sched[:-1]

def get_num_scheduler(x):
    patch_name = x[2].split('/')[-1].split('_')
    sched = []
    ind = 3
    while ':' in patch_name[patch_name.index('adv') + ind]:
        sched += list(map(int, patch_name[patch_name.index('adv') + ind].split(':')))
        ind += 1
    return sched

sort_dict = {
    'l2': lambda x: -float(x[-1]),
    'ap': lambda x: -float(x[0]),
    'fx': get_sg_coef,
    'fa': get_fa_coef,
    'lon': get_lo_num,
    'loc': get_lo_coef,
    'yolov': get_yolo_version,
    'sched': get_num_scheduler,
}

def get_sort_values(sort_key):
    keys = sort_key.split('_')
    return lambda x: tuple([sort_dict[key](x) for key in keys])

def plot_patches(to_plot, sort_key='l2', ncols=5, title='ap'):
    to_plot = sorted(to_plot, key=get_sort_values(sort_key))
    nrows = (len(to_plot) + ncols - 1) // ncols
    f, ax = plt.subplots(ncols=ncols, nrows=nrows, sharex=True, sharey=True, figsize=(3 * ncols + 1, len(to_plot) // ncols * 3 + 1))
    f.set_tight_layout(True)
    # f.patch.set_visible(False)
    for r in range(nrows):
        for c in range(ncols):
            ax[r, c].axis('off')
            if r * ncols + c >= len(to_plot):
                f.delaxes(ax[r, c])
                continue
            cur_to_plot = to_plot[r * ncols + c]
            img = cur_to_plot[2]
            ax[r, c].imshow(mpimg.imread(img))
            ttl = ''
            img_name = img.split('/')[-1][:-4].split('_')
            if 'ap' in title.split('_'):
                ttl += r"$\bf{" + '%.2f' % (100 * float(to_plot[r * ncols + c][0])) + "}$"
            sg = None
            if 'yolov' in title.split('_'):
                ttl += ' v' + str(get_yolo_version(cur_to_plot))
            if 'fx' in title.split('_') or 'fa' in title.split('_'):
                if 'fa' in img_name:
                    sg = img_name[img_name.index('fa') - 1]
                elif 'fx' in img_name:
                    sg = img_name[img_name.index('fx') + 1]
                if sg:
                    ttl += ' sg %.1E' % Decimal(float(sg))
            if 'fx' in title.split('_') and 'fx' in img_name:
                ttl += ' fx'
            if 'fa' in title.split('_') and 'fa' in img_name:
                fa_value = float(img_name[img_name.index('fa') + 1])
                ttl += ' fa_%.1E' % Decimal(fa_value * float(sg))
            if 'lo' in title.split('_') and 'lo' in img_name:
                ttl += ' lo_%d_%.1E' % (get_lo_num(cur_to_plot), get_lo_coef(cur_to_plot))
            if 'sched' in title.split('_'):
                ttl += ' ' + get_scheduler(cur_to_plot).replace('_', ' ')
            ax[r, c].title.set_text(ttl)
    plt.subplots_adjust(wspace=0, hspace=0.2)
    plt.savefig('tbd.png')
    
