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
def construct_guidance_dict(size_indices: list = None, size_values: list = None):  # TODO Support all modes
    return {"size_indices": size_indices, "size_values": size_values}


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

def register_to_layer(maps_recorder, cross_attn_layer):
    def replacement_forward(x, context=None):
        self_attn_marker = False

        h = cross_attn_layer.heads

        q = cross_attn_layer.to_q(x)
        if context is None:
            context = x
            self_attn_marker = True
        k = cross_attn_layer.to_k(context)
        v = cross_attn_layer.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * cross_attn_layer.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        maps_recorder.stored_q = q
        maps_recorder.stored_k = k
        maps_recorder.stored_v = v
        maps_recorder.self_attn_marker = self_attn_marker
        return cross_attn_layer.to_out(out)

    return replacement_forward


def self_guidance_loss(attn_maps: list, self_guidance_dict: dict):
    # Size losses
    size_indices = self_guidance_dict['size_indices']
    size_values = self_guidance_dict['size_values']
    assert len(size_values) == len(size_indices), 'OOPS, there should be an equal amount of values and indices'
    loss = 0
    for i, index in enumerate(size_indices):
        for attn_map in attn_maps:
            loss += l1_loss(size_fn(attn_map), size_values[i])

    return loss