import torch
import numpy as np

used_cross_attn_layers = ['CrossAttnDownBlock2D', 'CrossAttnUpBlock2D']
up_blocks_names = {"UpBlock2D", "CrossAttnUpBlock2D"}


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


def register_activation_recr(net_, controls, is_in_up=False):
    if net_.__class__.__name__ == 'ResnetBlock2D' and is_in_up:
        recorder = ActivationMapsRecorder()
        net_.register_activations_recorder(recorder)
        controls.append(recorder)
        return
    elif hasattr(net_, 'children'):
        for net__ in net_.children():
            new_flag = (is_in_up or (net_.__class__.__name__ in up_blocks_names))
            register_activation_recr(net__, controls, new_flag)

# Funcs to threshold the maps
def normalize_map(attn_map):
    return (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())


def threshold_map(attn_map, s=10.0):
    return normalize_map(torch.sigmoid(s * (normalize_map(attn_map) - 0.5)))


# Defining the g functions for different edits
def centroid_fn(relevant_att_map: torch.Tensor):  # shape: w, h, c
    w = torch.arange(relevant_att_map.shape[0]).to("cuda")
    h = torch.arange(relevant_att_map.shape[1]).to("cuda")
    map_mult_w = relevant_att_map * w
    map_mult_h = relevant_att_map * h
    return torch.cat([map_mult_w.flatten().sum(dim=0, keepdim=True), map_mult_h.flatten().sum(dim=0, keepdim=True)], 0) / relevant_att_map.sum()


def size_fn(relevant_att_map: torch.Tensor):
    return relevant_att_map.sum(axis=[0, 1]) / (relevant_att_map.shape[0] * relevant_att_map.shape[1])


def shape_fn(relevant_att_map: torch.Tensor, s=10.0):
    return threshold_map(relevant_att_map, s)


def appearance_fn(relevant_att_map: torch.Tensor, relevant_network_activations: torch.Tensor, threshold: float):
    shape = shape_fn(relevant_att_map, threshold)
    return (shape * relevant_network_activations).sum(dim=[0, 1]) / shape.sum()


# Functions for guidance
class MapsRecorder:
    def __init__(self):
        self.q = None
        self.k = None
        self.maps = None


class ActivationMapsRecorder:
    def __init__(self):
        self.recorded_maps = None
        self.recorded_appearance = None

def self_guidance_loss(attn_maps: list, self_guidance_dict: dict, initial_maps: list):
    loss = torch.zeros(1, device=attn_maps[0].device)
    for j, attn_map in enumerate(attn_maps):
        # resize map to h X w X tokens
        hw = int(np.sqrt(attn_map.shape[0]))
        rel_map = attn_map.reshape(hw, hw, attn_map.shape[-1])
        initial_map = initial_maps[j].reshape(hw, hw, attn_map.shape[-1])

        # Size losses
        if "size" in self_guidance_dict:
            size_weight = self_guidance_dict["size"]["weight"] if "weight" in self_guidance_dict["size"] else 1.0
            size_indices = self_guidance_dict['size']['indices']
            size_values = self_guidance_dict['size']['values']
            assert len(size_indices) == len(size_values), 'OOPS, there should be an equal amount of values and indices'

            for i, index in enumerate(size_indices):
                calc_size = size_fn(threshold_map(rel_map[:, :, index]))
                if self_guidance_dict['size']['mode'] == "absolute":
                    target_value = size_values[i]
                else:
                    target_value = size_fn(threshold_map(initial_map[:, :, index])) * size_values[i]
                loss += torch.abs(calc_size - target_value) * size_weight
        # Position losses
        if "position" in self_guidance_dict:
            position_weight = self_guidance_dict["position"]["weight"] if "weight" in self_guidance_dict["position"] else 1.0
            position_indices = self_guidance_dict['position']['indices']
            position_values = self_guidance_dict['position']['values']

            assert len(position_indices) == len(position_values), 'OOPS, there should be an equal amount of values and indices'

            for i, index in enumerate(position_indices):
                calc_position = centroid_fn(threshold_map(rel_map[:, :, index]))
                if self_guidance_dict['position']['mode'] == "absolute":
                    target_value = position_values[i]
                else:
                    target_value = centroid_fn(threshold_map(initial_map[:, :, index])) + position_values[i]
                loss += torch.abs(calc_position - target_value).mean() * position_weight
        # Shape losses
        if "shape" in self_guidance_dict:
            shape_weight = self_guidance_dict["shape"]["weight"] if "weight" in self_guidance_dict["shape"] else 1.0
            shape_indices = self_guidance_dict['shape']['indices']

            for i, index in enumerate(shape_indices):
                actual_shape = shape_fn(rel_map[:, :, index])
                desired_shape = shape_fn(initial_map[:, :, index])

                loss += torch.abs(actual_shape - desired_shape).mean() * shape_weight
        # Appearance losses
        if "appearance" in self_guidance_dict:
            pass
    return loss
