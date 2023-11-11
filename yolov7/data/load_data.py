import fnmatch
import math
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple


class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


class MaxProbExtractor(nn.Module):
    """ MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id, num_cls, func, name):
        super(MaxProbExtractor, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        # self.config = config
        if func is None:
            self.func = lambda obj, cls: obj
        else:
            self.func = func
        self.name = name

    def forward(self, YOLOoutput):
        # get values necessary for transformation
        if self.name == 'yolov2':
            if YOLOoutput.dim() == 3:
                YOLOoutput = YOLOoutput.unsqueeze(0)
            batch = YOLOoutput.size(0)
            assert (YOLOoutput.size(1) == (5 + self.num_cls) * 5)
            h = YOLOoutput.size(2)
            w = YOLOoutput.size(3)
            # transform the output tensor from [batch, 425, 19, 19] to [batch, 80, 1805]
            output = YOLOoutput.view(batch, 5, 5 + self.num_cls, h * w)  # [batch, 5, 85, 361]
            output = output.transpose(1, 2).contiguous()  # [batch, 85, 5, 361]
            output = output.view(batch, 5 + self.num_cls, 5 * h * w)  # [batch, 85, 1805]
            # output_objectness = torch.sigmoid(output[:, 4, :])  # [batch, 1805]
            output_objectness = output[:, 4, :]  # [batch, 1805]
            output = output[:, 5:5 + self.num_cls, :]  # [batch, 80, 1805]
            # perform softmax to normalize probabilities for object classes to [0,1]
            normal_confs = torch.nn.Softmax(dim=1)(output)
            # we only care for probabilities of the class of interest (person)
            confs_for_class = normal_confs[:, self.cls_id, :]
            # confs_if_object = output_objectness #confs_for_class * output_objectness
            # confs_if_object = confs_for_class * output_objectness
            # # confs_if_object = self.config.loss_target(output_objectness, confs_for_class)
            confs_if_object = self.func(output_objectness, confs_for_class)
            # find the max probability for person
            # max_conf, max_conf_idx = torch.max(confs_if_object, dim=1)
            # return max_conf
            return [confs_if_object]

        else:
            raise ValueError


class NPSCalculator(nn.Module):
    """NMSCalculator: calculates the non-printability score of a patch.

    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.

    """

    def __init__(self, printability_file):
        super(NPSCalculator, self).__init__()
        printability_array = self.get_printability_array(printability_file, 1).unsqueeze(1).detach()
        self.register_buffer('printability_array', printability_array)

    def forward(self, adv_patch):
        # calculate euclidian distance between colors in patch and colors in printability_array
        # square root of sum of squared difference
        if adv_patch.dim == 3:
            adv_patch = adv_patch.unsqueeze(0)

        adv_patch.unsqueeze(0)
        color_dist = (adv_patch - self.printability_array + 0.000001)
        color_dist = color_dist ** 2
        color_dist = torch.sum(color_dist, 2) + 0.000001
        # color_dist = torch.sum(color_dist, 1)+0.000001
        color_dist = torch.sqrt(color_dist)
        # only work with the min distance
        color_dist_prod = torch.min(color_dist, 0)[0]  # test: change prod for min (find distance to closest color)
        # calculate the nps by summing over all pixels
        # nps_score = torch.sum(color_dist_prod, 0)
        # nps_score = torch.sum(nps_score, 0)
        nps_score = color_dist_prod.sum()
        return nps_score/torch.numel(adv_patch)

    def get_printability_array(self, printability_file, side):
        printability_list = []

        # read in printability triplets and put them in a list
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))

        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((side, side), red))
            printability_imgs.append(np.full((side, side), green))
            printability_imgs.append(np.full((side, side), blue))
            printability_array.append(printability_imgs)

        printability_array = np.asarray(printability_array)
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array)
        return pa


class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total variation (TV) of an adversarial patch.

    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch):
        # bereken de total variation van de adv_patch
        if adv_patch.dim == 3:
            adv_patch = adv_patch.unsqueeze(0)
        tvcomp1 = torch.abs(adv_patch[:, :, :, 1:] - adv_patch[:, :, :, :-1] + 0.000001).sum()
        tvcomp2 = torch.abs(adv_patch[:, :, 1:, :] - adv_patch[:, :, :-1, :] + 0.000001).sum()

        # tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, :, 1:] - adv_patch[:, :, :, :-1]+0.000001), 0)
        # tvcomp1 = torch.sum(torch.sum(tvcomp1, 0), 0)
        # tvcomp2 = torch.sum(torch.abs(adv_patch[:, :, 1:, :] - adv_patch[:, :, :-1, :]+0.000001), 0)
        # tvcomp2 = torch.sum(torch.sum(tvcomp2, 0), 0)
        tv = tvcomp1 + tvcomp2
        return tv / torch.numel(adv_patch)


class PatchTransformer(nn.Module):
    """PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.

    """

    def __init__(self):
        super(PatchTransformer, self).__init__()
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
        self.minangle = -20 / 180 * math.pi
        self.maxangle = 20 / 180 * math.pi
        self.medianpooler = MedianPool2d(7, same=True)

        ksize = 5
        half = (ksize - 1) * 0.5
        sigma = 0.3 * (half - 1) + 0.8
        x = np.arange(-half, half + 1)
        x = np.exp(- np.square(x / sigma) / 2)
        x = np.outer(x, x)
        x = x / x.sum()
        x = torch.from_numpy(x).float()
        kernel = torch.zeros(3, 3, ksize, ksize)
        for i in range(3):
            kernel[i, i] = x
        self.register_buffer('kernel', kernel)
        '''
        kernel = torch.cuda.FloatTensor([[0.003765, 0.015019, 0.023792, 0.015019, 0.003765],                                                                                    
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],                                                                                    
                                         [0.023792, 0.094907, 0.150342, 0.094907, 0.023792],                                                                                    
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],                                                                                    
                                         [0.003765, 0.015019, 0.023792, 0.015019, 0.003765]])
        self.kernel = kernel.unsqueeze(0).unsqueeze(0).expand(3,3,-1,-1)
        # It's wrong!
        '''
    def forward(self, adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=True, lc_scale=0.1, pooling='median', rand_sub=False, old_fasion=True):
        if adv_patch.dim() == 3:
            adv_patch = adv_patch.unsqueeze(0)

        B, L, _ = lab_batch.shape
        _, C, H, W = adv_patch.shape
        SBS = B * L

        # add pooling
        # adv_patch = F.conv2d(adv_patch.unsqueeze(0),self.kernel,padding=(2,2))
        if pooling == 'median':  # kb: is -> == for strings here and below
            adv_patch = self.medianpooler(adv_patch)
        elif pooling == 'avg':
            adv_patch = F.avg_pool2d(adv_patch, 7, 3)
        elif pooling == 'gauss':
            adv_patch = F.conv2d(adv_patch, self.kernel, padding=2)
        elif pooling is not None:
            raise ValueError
        # Determine size of padding
        # pad_w = (img_size - adv_patch.size(-2)) / 2
        # pad_h = (img_size - adv_patch.size(-1)) / 2
        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(1)  # .unsqueeze(0)
        adv_batch = adv_patch.expand(B, L, -1, -1, -1)
        batch_size = torch.Size((B, L))

        # Contrast, brightness and noise transforms

        # Create random contrast tensor
        # contrast = torch.cuda.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
        contrast = adv_patch.new(batch_size).uniform_(self.min_contrast, self.max_contrast)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        # contrast = contrast.to(adv_patch)

        # Create random brightness tensor
        # brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
        brightness = adv_patch.new(batch_size).uniform_(self.min_brightness, self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        # brightness = brightness.to(adv_patch)

        # Create random noise tensor
        # noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor
        noise = adv_patch.new(adv_batch.shape).uniform_(-1, 1) * self.noise_factor

        # Apply contrast/brightness/noise, clamp
        adv_batch = adv_batch * contrast + brightness + noise
        adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)

        # Where the label class_id is 1 we don't want a patch (padding) --> fill mask with zero's
        # cls_ids = torch.narrow(lab_batch, 2, 0, 1)
        # cls_mask = cls_ids.expand(-1, -1, 3)
        # cls_mask = cls_mask.unsqueeze(-1)
        # cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))
        # cls_mask = cls_mask.unsqueeze(-1)
        # cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))
        # # msk_batch = torch.cuda.FloatTensor(cls_mask.size()).fill_(1) - cls_mask
        # msk_batch = adv_patch.new(cls_mask.size()).fill_(1) + cls_mask  # (B, L, 3, 324, 324)
        msk_batch = adv_patch.new(adv_batch.shape).fill_(1).logical_and((lab_batch[:, :, 0] == 0).view(B, L, 1, 1, 1))

        # Pad patch and mask to image dimensions
        # mypad = nn.ConstantPad2d((int(pad + 0.5), int(pad), int(pad + 0.5), int(pad)), 0)
        # mypad = nn.ConstantPad2d((int(pad_h + 0.5), int(pad_h), int(pad_w + 0.5), int(pad_w)), 0)
        # adv_batch = mypad(adv_batch)
        # msk_batch = mypad(msk_batch)

        # Rotation and rescaling transforms
        anglesize = (lab_batch.size(0) * lab_batch.size(1))
        if do_rotate:
            # angle = torch.cuda.FloatTensor(anglesize).uniform_(self.minangle, self.maxangle)
            angle = adv_patch.new(anglesize).uniform_(self.minangle, self.maxangle)
        else:
            # angle = torch.cuda.FloatTensor(anglesize).fill_(0)
            angle = adv_patch.new(anglesize).fill_(0)

        # Resizes and rotates
        # current_patch_size = adv_patch.size(-1) if adv_patch.size(-1) == adv_patch.size(-2) else math.sqrt(adv_patch.size(-1) * adv_patch.size(-2))
        # lab_batch_scaled = adv_patch.new(lab_batch.size()).fill_(0)
        # lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * img_size
        # lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * img_size
        # lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * img_size
        # lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * img_size
        target_size = torch.sqrt(((lab_batch[:, :, 3].mul(0.2)) ** 2) + ((lab_batch[:, :, 4].mul(0.2)) ** 2))
        target_x = lab_batch[:, :, 1].view(np.prod(batch_size))
        target_y = lab_batch[:, :, 2].view(np.prod(batch_size))
        targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size))
        targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size))
        if rand_loc:
            off_x = targetoff_x * (adv_patch.new(targetoff_x.size()).uniform_(-lc_scale, lc_scale))
            target_x = target_x + off_x
            off_y = targetoff_y * (adv_patch.new(targetoff_y.size()).uniform_(-lc_scale, lc_scale))
            target_y = target_y + off_y

        if old_fasion:
            target_y = target_y - 0.05
        else:
            target_y = target_y - 0.15 * targetoff_y

        scale = target_size
        # scale = target_size / current_patch_size
        scale = scale.view(anglesize)

        adv_batch = adv_batch.view(SBS, C, H, W)
        msk_batch = msk_batch.view(SBS, C, H, W)

        if rand_sub is True:
            width = adv_batch.new(size=[SBS, 1]).uniform_(0.5, 1)
            height = adv_batch.new(size=[SBS, 1]).uniform_(0.8, 1)
            wst = adv_batch.new(size=[SBS, 1]).uniform_(0, 1) * (1 - width)
            hst = adv_batch.new(size=[SBS, 1]).uniform_(0, 1) * (1 - height)
            W_msk = torch.arange(W, device=adv_batch.device).expand(SBS, W) < (wst * W)
            W_msk.logical_xor_(torch.arange(W, device=adv_batch.device).expand(SBS, W) < ((wst + width) * W))
            W_msk = W_msk.view(SBS, 1, 1, W)
            H_msk = torch.arange(H, device=adv_batch.device).expand(SBS, H) < (hst * H)
            H_msk.logical_xor_(torch.arange(H, device=adv_batch.device).expand(SBS, H) < ((hst + height) * H))
            H_msk = H_msk.view(SBS, 1, H, 1)
            msk_batch = msk_batch.logical_and(W_msk.logical_and(H_msk))

        tx = (-target_x + 0.5) * 2
        ty = (-target_y + 0.5) * 2
        sin = torch.sin(angle).to(adv_patch)
        cos = torch.cos(angle).to(adv_patch)

        # Theta = rotation,rescale matrix
        # theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        theta = adv_patch.new(anglesize, 2, 3).fill_(0)
        theta[:, 0, 0] = cos / scale
        theta[:, 0, 1] = sin / scale
        theta[:, 0, 2] = tx * cos / scale + ty * sin / scale
        theta[:, 1, 0] = -sin / scale
        theta[:, 1, 1] = cos / scale
        theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale

        # b_sh = adv_batch.shape
        # grid = F.affine_grid(theta, adv_batch.shape)
        grid = F.affine_grid(theta, [SBS, C, img_size, img_size])

        adv_batch_t = F.grid_sample(adv_batch, grid, align_corners=False)
        msk_batch_t = F.grid_sample(msk_batch.to(adv_batch), grid, align_corners=False)

        '''
        # Theta2 = translation matrix
        theta2 = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        theta2[:, 0, 0] = 1
        theta2[:, 0, 1] = 0
        theta2[:, 0, 2] = (-target_x + 0.5) * 2
        theta2[:, 1, 0] = 0
        theta2[:, 1, 1] = 1
        theta2[:, 1, 2] = (-target_y + 0.5) * 2

        grid2 = F.affine_grid(theta2, adv_batch.shape)
        adv_batch_t = F.grid_sample(adv_batch_t, grid2)
        msk_batch_t = F.grid_sample(msk_batch_t, grid2)

        '''
        adv_batch_t = adv_batch_t.view(B, L, C, img_size, img_size)
        msk_batch_t = msk_batch_t.view(B, L, C, img_size, img_size)

        adv_batch_t = torch.clamp(adv_batch_t, 0.000001, 0.999999)
        #img = msk_batch_t[0, 0, :, :, :].detach().cpu()
        #img = transforms.ToPILImage()(img)
        #img.show()
        #exit()

        return adv_batch_t * msk_batch_t


class PatchApplier(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.

    """

    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, adv_batch):
        advs = torch.unbind(adv_batch, 1)
        for adv in advs:
            img_batch = torch.where((adv == 0), img_batch, adv)
        return img_batch


def collate_wo_max_label(batch):
    images = torch.stack([el[0] for el in batch], 0)
    labels = torch.cat([torch.cat([torch.full((el[1].shape[0], 1), i), el[1]], 1) for i, el in enumerate(batch)], 0)
    # print(batch[0][2])
    # 1/0
    return images, labels, torch.stack([el[2] for el in batch], 0)


class InriaDataset(Dataset):
    """InriaDataset: representation of the INRIA person dataset.

    Internal representation of the commonly used INRIA person dataset.
    Available at: http://pascal.inrialpes.fr/data/human/

    Attributes:
        len: An integer number of elements in the
        img_dir: Directory containing the images of the INRIA dataset.
        lab_dir: Directory containing the labels of the INRIA dataset.
        img_names: List of all image file names in img_dir.
        shuffle: Whether to shuffle the dataset.

    """

    def __init__(self, img_dir, lab_dir, imgsize, shuffle=True):
        n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images = n_png_images + n_jpg_images
        if lab_dir is not None:
            if not isinstance(lab_dir, list):
                lab_dir = [lab_dir]
            n_labels = len(fnmatch.filter(os.listdir(lab_dir[0]), '*.txt'))
            assert n_images == n_labels, "Number of images and number of labels don't match"
        self.len = n_images
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.imgsize = imgsize
        self.img_names = fnmatch.filter(os.listdir(img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
        self.shuffle = shuffle
        self.img_paths = []
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
        self.lab_paths = []
        if self.lab_dir is None:
            lab_dir = ['']
        for img_name in self.img_names:
            lab_path = [os.path.join(ld, img_name).replace('.jpg', '.txt').replace('.png', '.txt') for ld in lab_dir]
            self.lab_paths.append(lab_path)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        # img_path = os.path.join(self.img_dir, self.img_names[idx])
        img_path = self.img_paths[idx]
        # lab_path = os.path.join(self.lab_dir, self.img_names[idx]).replace('.jpg', '.txt').replace('.png', '.txt')
        lab_path = self.lab_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.lab_dir is not None:
            label = []
            for lp in lab_path:
                if os.path.getsize(lp):       # check to see if label file contains data.
                    lb = torch.from_numpy(np.loadtxt(lp)).float()
                    if lb.dim() == 1:
                        lb = lb.unsqueeze(0)
                    label.append(lb)
                else:
                    label.append(torch.ones([1, 5]).float())
        else:
            label = None

        image, label = self.pad_and_scale(image, label)
        transform = transforms.ToTensor()
        image = transform(image)
        if self.lab_dir is not None:
            label_padded = self.pad_lab(label)
            if len(label) == 1:
                label, label_padded = label[0], label_padded[0]
            # print('here', image.shape, label.shape, label_padded.shape)
            return image, label, label_padded
        else:
            return image, lab_path[0]

    def pad_and_scale(self, img, label):
        w, h = img.size
        if w == h:
            padded_img = img
        else:
            dim_to_pad = 1 if w < h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
                padded_img.paste(img, (int(padding), 0))
                if label is not None:
                    for i in range(len(label)):
                        label[i][:, [1]] = (label[i][:, [1]] * w + padding) / h
                        label[i][:, [3]] = (label[i][:, [3]] * w / h)
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
                padded_img.paste(img, (0, int(padding)))
                if label is not None:
                    for i in range(len(label)):
                        label[i][:, [2]] = (label[i][:, [2]] * h + padding) / w
                        label[i][:, [4]] = (label[i][:, [4]] * h / w)
        resize = transforms.Resize((self.imgsize, self.imgsize))
        padded_img = resize(padded_img)     # choose here
        return padded_img, label

    def pad_lab(self, label):
        padded_lab = []
        for lab in label:
            pad_size = 15 - lab.shape[0]  # todo: maybe go pack to max_n_labels
            padded_lab.append(F.pad(lab, (0, 0, 0, pad_size), value=-1) if pad_size > 0 else lab)
        return padded_lab
