from typing import Optional

from mmdet.models import YOLOV3, DETECTORS

from .i_detector import IDetector


@DETECTORS.register_module()
class YOLOV3Detector(YOLOV3, IDetector):
    """Decorator to set some patch forward functions."""

    def extract_feat(self, img):
        return super().extract_feat(img)

    def get_bboxes(self, feat, img_metas, rescale=False):
        return self.bbox_head.simple_test(feat, img_metas, rescale=rescale)

    def forward_test(self, img, img_metas=None, rescale=True):
        if img_metas is None:
            scale = img.shape[2]
            scale = (scale, scale, scale, scale)
            img_metas = [dict(scale_factor=scale) for _ in range(len(img))]
        return self.get_bboxes(self.extract_feat(img), img_metas, rescale)
