import os
import torch
import itertools
import yaml
import numpy as np
import matplotlib.pyplot as plt
import argparse
import matplotlib
import glob
from PIL import Image

from tqdm import tqdm
from scipy.interpolate import interp1d
from torchvision import transforms

unloader = transforms.ToPILImage()
matplotlib.use('Agg')

from yolov7.utils.general import check_file
from yolo2 import load_data
from yolo2 import utils
from yolov7.utils.torch_utils import TPSGridGen
from yolov7.load_adv import get_model


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--net', default='yolov2', help='target net name')
parser.add_argument('--batch-size', type=int, default=64, help='batch size')
parser.add_argument('--device', default='cuda:0', help='')
parser.add_argument('--suffix', default='tbd', help='suffix name')
parser.add_argument('--prepare-data', default=False, action='store_true', help='')
parser.add_argument('--load-path', default=None, help='load patch')
parser.add_argument('--mask', type=str, default=None, help='ex: adv_500*')
pargs = parser.parse_args()


device = torch.device(pargs.device)

imgsz = 416  # standard for Inria
with open(check_file('data/inria.yaml')) as f:
    data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
darknet_model = get_model(data_dict, device, pargs.net)

class_names = data_dict['names']

target_func = lambda obj, cls: obj
patch_applier = load_data.PatchApplier().to(device)
patch_transformer = load_data.PatchTransformer().to(device)
prob_extractor = load_data.MaxProbExtractor(0, 80, target_func, pargs.net).to(device)
total_variation = load_data.TotalVariation().to(device)

target_control_points = torch.tensor(list(itertools.product(
    torch.arange(-1.0, 1.00001, 2.0 / 4),
    torch.arange(-1.0, 1.00001, 2.0 / 4),
)))

tps = TPSGridGen(torch.Size([300, 300]), target_control_points)
tps.to(device)

target_func = lambda obj, cls: obj
prob_extractor = load_data.MaxProbExtractor(0, 80, target_func, pargs.net).to(device)

results_dir = './results/result_' + pargs.suffix

if pargs.prepare_data:
    conf_thresh = 0.5
    nms_thresh = 0.4
    img_ori_dir = os.getcwd() + '/yolov7/data/INRIAPerson/Test/pos'
    img_dir = os.getcwd() + '/yolov7/data/test_padded'
    lab_dir = os.getcwd() + '/yolov7/data/test_lab_%s' % pargs.net
    data_nl = load_data.InriaDataset(img_ori_dir, None, int(data_dict['max_lab']), int(data_dict['img_size']), shuffle=False)
    loader_nl = torch.utils.data.DataLoader(data_nl, batch_size=pargs.batch_size, shuffle=False, num_workers=10)
    if lab_dir is not None:
        if not os.path.exists(lab_dir):
            os.makedirs(lab_dir)
    if img_dir is not None:
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
    print('preparing the test data')
    with torch.no_grad():
        for batch_idx, (data, labs) in tqdm(enumerate(loader_nl), total=len(loader_nl)):
            data = data.to(device)
            output = darknet_model(data)
            all_boxes = utils.get_region_boxes_general(output, darknet_model, conf_thresh, pargs.net)
            for i in range(data.size(0)):
                boxes = all_boxes[i]
                boxes = utils.nms(boxes, nms_thresh)
                new_boxes = boxes[:, [6, 0, 1, 2, 3]]
                new_boxes = new_boxes[new_boxes[:, 0] == 0]
                new_boxes = new_boxes.detach().cpu().numpy()
                if lab_dir is not None:
                    save_dir = os.path.join(lab_dir, labs[i])
                    np.savetxt(save_dir, new_boxes, fmt='%f')
                    img = unloader(data[i].detach().cpu())
                if img_dir is not None:
                    save_dir = os.path.join(img_dir, labs[i].replace('.txt', '.png'))
                    img.save(save_dir)
    print('preparing done')

img_dir_test = os.getcwd() + '/yolov7/data/test_padded'
lab_dir_test = os.getcwd() + '/yolov7/data/test_lab_%s' % pargs.net
test_data = load_data.InriaDataset(img_dir_test, lab_dir_test, int(data_dict['max_lab']), int(data_dict['img_size']), shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=pargs.batch_size, shuffle=False, num_workers=10)
loader = test_loader
epoch_length = len(loader)
print(f'One epoch is {len(loader)}')


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
         old_fasion=True):
    model.eval()
    total = 0.0
    batch_num = len(loader)

    with torch.no_grad():
        positives = []
        for batch_idx, (data, target) in enumerate(loader):
            data = data.to(device)

            if adv_patch is not None:
                target = target.to(device)
                adv_batch_t = patch_transformer(adv_patch, target, int(data_dict['img_size']), do_rotate=True, rand_loc=False,
                                                pooling='median', old_fasion=old_fasion)
                data = patch_applier(data, adv_batch_t)
            if batch_idx == 0:
                patched_sample = transforms.ToPILImage()(np.uint8((data[0] * 255).permute(1, 2, 0).detach().cpu().numpy()))
                patched_sample.save('tbd.png')
            output = model(data)
            all_boxes = utils.get_region_boxes_general(output, model, conf_thresh, pargs.net)
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

save_dir = './test_results'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, pargs.suffix)

plt.figure(figsize=[10, 10])
if pargs.load_path is not None:
    if os.path.isdir(pargs.load_path):
        if pargs.mask is None:
            img_paths = glob.glob(f'{pargs.load_path}/*.png') + glob.glob(f'{pargs.load_path}/*.npy')
        else:
            img_paths = glob.glob(f'{pargs.load_path}/{pargs.mask}')
        print(img_paths)
    else:
        img_paths = [pargs.load_path]
    cmap = plt.cm.coolwarm(np.linspace(0, 1, len(img_paths)))
    res = []
    for i, img_path in tqdm(enumerate(img_paths), total=len(img_paths)):
        try:
            patch = torch.from_numpy(np.load(img_path)[:1]).to(device)
        except ValueError:
            patch = Image.open(img_path)
            patch = transforms.ToTensor()(patch).to(device)

        test_patch = patch.detach().clone()
        prec, rec, ap, confs = test(darknet_model, test_loader, adv_patch=test_patch, conf_thresh=0.01, old_fasion=True)
        res.append((img_path, prec, rec, ap))
        np.savez(save_path, prec=prec, rec=rec, ap=ap, confs=confs, adv_patch=test_patch.detach().cpu().numpy())
    
    res = sorted(res, key=lambda x: -x[-1])
    for i in range(len(res)):
        img_path, prec, rec, ap = res[i]
        print('AP is %.4f'% ap, f'for {img_path[:-4]}')
        plt.plot(rec, prec, color=cmap[i], label=img_path.split('.')[0].split('/')[-1] + ': ap %.3f' % ap)

else:
    prec, rec, ap, confs = test(darknet_model, test_loader, conf_thresh=0.01, old_fasion=True)
    print('AP is %.4f'% ap)
    plt.plot(rec, prec)
    leg = [pargs.suffix + ': ap %.3f' % ap]

plt.plot([0, 1], [0, 1], 'k--')
plt.legend(loc=4)
plt.title('PR-curve')
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.ylim([0, 1.05])
plt.xlim([0, 1.05])
plt.savefig(os.path.join(save_dir, f'PR-curve_{pargs.suffix}.png'), dpi=300)
