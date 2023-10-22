import torch
import numpy as np

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


# Funcs to threshold the maps
def normalize_map(attn_map):
    return (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())


def threshold_map(attn_map, s=10.0):
    return normalize_map(torch.sigmoid(s * (normalize_map(attn_map) - 0.5)))


# Defining the g functions for different edits
def centroid_fn(relevant_att_map: torch.Tensor):  # shape: w, h, c
    w = torch.arange(relevant_att_map.shape[0])
    h = torch.arange(relevant_att_map.shape[1])
    map_mult_w = relevant_att_map * w
    map_mult_h = relevant_att_map * h
    return torch.cat([map_mult_w.sum(dim=[0, 1]), map_mult_h.sum(dim=[0, 1])], 0) / relevant_att_map.sum()


def size_fn(relevant_att_map: torch.Tensor):
    return relevant_att_map.sum(axis=[0, 1]) / (relevant_att_map.shape[0] * relevant_att_map.shape[1])


def shape_fn(relevant_att_map: torch.Tensor, s=10.0):
    return threshold_map(relevant_att_map, s)


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
            # resize map to h X w X tokens
            hw = int(np.sqrt(attn_map.shape[0]))

            rel_map = attn_map.reshape(hw, hw, attn_map.shape[-1])

            calc_size = size_fn(threshold_map(rel_map[:, :, index]))
            loss += torch.abs(calc_size - size_coefs[i] * calc_size)

    return loss
