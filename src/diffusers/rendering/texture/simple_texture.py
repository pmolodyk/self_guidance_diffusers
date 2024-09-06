import logging
from math import ceil
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn import Module
from typing import Optional
from random import randint

from src.diffusers.models.tv_loss import total_variation
from src.diffusers.utils.typing_utils import _pair, _size_2_t
from src.diffusers.rendering.texture.i_texture import ITexture


class SimpleTexture(ITexture):
    """Simple texture generation.

    Args:
        fig_size: Output texture size.
        map_size: Original texture map size. If None, set equal to fig_size.
        color_transform: Color transformation.
    """

    def __init__(
        self,
        fig_size: _size_2_t,
        map_size: Optional[_size_2_t] = None,
        color_transform: Module = nn.Identity(),
        tile: bool = False,
        tile_size: int = 128,
        eval: bool = False,
    ) -> None:
        super().__init__()
        self.fig_size = _pair(fig_size)
        if map_size is None:
            self.map_size = self.fig_size
            self.do_interpolate = False
        else:
            self.map_size = _pair(map_size)
            self.do_interpolate = True
        self.color_transform = color_transform
        # self.tex_map = nn.Parameter(torch.empty((1, 3) + self.map_size)).to(device)
        # nn.init.constant_(self.tex_map, 0.5)  # Init as gray
        # self.tex_map = torch.Tensor(torch.empty((1, 3) + self.map_size)).to(device)
        self.tex_map = None
        self._WARNED_PATCH_NAN_OR_INF = False
        self.tile = tile
        self.tile_size = tile_size
        self.eval = eval

    def forward(
        self,
        tex_map: Optional[Tensor] = None,
        transform_color=True,
    ):
        if tex_map is None:
            tex_map = self.tex_map
        if self.do_interpolate:
            if self.tile:
                base_tile = F.interpolate(tex_map, self.tile_size)
                mults = [ceil(self.fig_size[-2 + i] / self.tile_size) + 1 for i in range(2)]
                patch_tiled = torch.tile(base_tile, mults)
                if self.eval:
                    init_x, init_y = 0, 0
                else:
                    init_x, init_y = randint(0, self.tile_size - 1), randint(0, self.tile_size - 1)
                tex = patch_tiled[:, :, init_x:init_x + self.fig_size[-2], 
                                        init_y:init_y + self.fig_size[-1]]
            else:
                tex = F.interpolate(tex_map, self.fig_size)
        else:
            tex = tex_map
        if transform_color:
            tex = self.color_transform(tex)
        return tex

    def tv_loss(self):
        return total_variation(self.tex_map).clamp_min(0.04)

    def loss(self):
        return self.tv_loss()

    def clamp_(self, clamp_shift=0.0):
        if not self._WARNED_PATCH_NAN_OR_INF and not torch.all(
            torch.isfinite(self.tex_map)
        ):
            logging.warning("NaN or Inf found in texture.")
            self._WARNED_PATCH_NAN_OR_INF = True
        self.tex_map.nan_to_num_(0, 0, 0)
        self.tex_map.clamp_(clamp_shift, 1 - clamp_shift)
