import torch
from einops import rearrange, repeat
from torch import einsum
from torch.nn.functional import l1_loss

used_cross_attn_layers = ['CrossAttnDownBlock2D', 'CrossAttnUpBlock2D']


def register_attention_layers_recr(net_, controls):
    # print(net_.__class__.__name__)
    if net_.__class__.__name__ == 'Attention':
        recorder = MapsRecorder()
        net_.register_attn_recorder(recorder)
        controls.append(recorder)
        return
    elif hasattr(net_, 'children'):
        for net__ in net_.children():
            register_attention_layers_recr(net__, controls)


# Parse inputs to construct self-guidance component
def construct_guidance_dict(size_indices: list = None, size_coefs: list = None):  # TODO Support all modes
    return {"size_indices": size_indices, "size_coefs": size_coefs}


# Defining the g functions for different edits
def centroid_fn(relevant_att_map: torch.Tensor):  # shape: w, h, c
    w = torch.arange(relevant_att_map.shape[0])
    h = torch.arange(relevant_att_map.shape[1])
    map_mult_w = relevant_att_map * w
    map_mult_h = relevant_att_map * h
    return torch.cat([map_mult_w.sum(dim=[0, 1]), map_mult_h.sum(dim=[0, 1])], 0) / relevant_att_map.sum()


def size_fn(relevant_att_map: torch.Tensor):
    w = relevant_att_map.shape[0]
    h = relevant_att_map.shape[1]
    return relevant_att_map.sum(dim=[0, 1]) / (h * w)


def shape_fn(relevant_att_map: torch.Tensor, threshold: float):
    return relevant_att_map * (relevant_att_map > threshold)


def appearance_fn(relevant_att_map: torch.Tensor, relevant_network_activations: torch.Tensor, threshold: float):
    shape = shape_fn(relevant_att_map, threshold)
    return (shape * relevant_network_activations).sum(dim=[0, 1]) / shape.sum()


# Functions for guidance
def size_loss_function(desired_size: float, actual_size: float):
    return abs(desired_size - actual_size)


def appearance_loss_function(desired_appearance: torch.Tensor, actual_appearance: torch.Tensor):
    pass


class MapsRecorder:
    def __init__(self):
        self.q = None
        self.k = None
        self.maps = None


def self_guidance_loss(attn_maps: list, self_guidance_dict: dict):
    # Size losses
    size_indices = self_guidance_dict['size_indices']
    size_coefs = self_guidance_dict['size_coefs']
    assert len(size_indices) == len(size_coefs), 'OOPS, there should be an equal amount of coefs and indices'
    loss = torch.zeros(1, device='cuda')
    for i, index in enumerate(size_indices):
        for attn_map in attn_maps:
            calc_size = size_fn(attn_map[:, :, index])
            loss += torch.abs(calc_size - size_coefs[i] * calc_size)

    return loss
