from torch.nn import Module
from torch import Tensor

from src.diffusers.utils.typing_utils import _pair, _size_2_t
from src.diffusers.rendering.texture.camou_texture import CamouflageTexture
from src.diffusers.rendering.texture.jitter_texture import JitterTexture


class JitterCamouTexture(CamouflageTexture):
    def __init__(
        self,
        colors: list[list[float]],
        fig_size: _size_2_t,
        color_transform: Module = ...,
        num_points=1,
        seed_ratio=0.7,
        resolution=4,
        blur=1,
        scale=1.1,
    ) -> None:
        super().__init__(
            colors, fig_size, color_transform, num_points, seed_ratio, resolution, blur
        )
        self.aux_jitter = JitterTexture(fig_size, scale)

    def forward(
        self,
        points: Tensor or None = None,
        tau=0.3,
        determinate=False,
        transform_color=True,
        is_train=True,
    ):
        tex = super().forward(points, tau, determinate, transform_color)
        return self.aux_jitter.forward(tex, transform_color, is_train)
