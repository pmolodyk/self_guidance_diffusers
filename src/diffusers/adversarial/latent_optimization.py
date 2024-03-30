import torch 
from src.diffusers.adversarial.load_target_model import get_adv_imgs
from yolo2.utils import get_det_loss
from tqdm import tqdm
import os

modes = {
    "max": torch.argmax,
    "min": torch.argmin
}

def latent_optimization(adv_model, detector, pipe, latents, adv_dataloader, device, pipeline, renderer, mode):
    N, c_l, h_l, w_l = latents.shape
    assert c_l == 4 and h_l == 32 and w_l == 32
    assert pipeline == '3d'
    assert mode in modes.keys()

    l2s_name = f"{adv_model}_los_{latents.shape[0]}.pt"
    if os.path.exists(l2s_name):
        l2s = torch.load(l2s_name)
        return modes[mode](l2s.mean(-1)).item()

    no_resamlping = 0  # so that there is no resampling in rendering
    
    l2s = []

    for latent in tqdm(latents):
        l2s.append([])
        for (imgs, targets_all) in adv_dataloader:
            # print("imgs.shape", imgs.shape)
            cur_latent = latent.detach().requires_grad_(True)[None]
            adv_patch = pipe.vae.decode(cur_latent / pipe.vae.config.scaling_factor, return_dict=False)[0]
            imgs = imgs.to(device, non_blocking=True)
            adv_imgs, targets_padded = get_adv_imgs(adv_patch, pipeline, targets_all, None, None,
                                                    None, imgs, renderer, no_resamlping, adv_dataloader)
            no_resamlping = 1
            loss, valid_num = get_det_loss(detector, adv_imgs, targets_padded.to(device), name=adv_model, mode='max')
            if valid_num > 0:
                loss = loss / valid_num
            else:
                loss = 0
            grads = torch.autograd.grad(loss, cur_latent)
            l2s[-1].append(grads[0].detach().cpu().norm(2))
    
    l2s = torch.Tensor(l2s)
    torch.save(l2s, l2s_name)
    return modes[mode](l2s.mean(-1)).item()
