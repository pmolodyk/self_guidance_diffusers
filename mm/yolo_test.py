import torch
from os import path as osp
from .yolov3 import YOLOV3Detector


configs_mmdet = {
    "yolov3": {
        "weight": "data/yolov3_d53_mstrain-416_273e_coco-2b60fcd9.pth",
        "model": YOLOV3Detector,
        "cfg": "mm/configs/yolov3_d53_mstrain-416_273e_coco.py",
        "weights": "mm/yolov3_d53_mstrain-416_273e_coco-2b60fcd9.pth",
        "weights_http": "https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-416_273e_coco/yolov3_d53_mstrain-416_273e_coco-2b60fcd9.pth"
    }
}
    # label_dir: ~/dataset/INRIAPerson/custom_ann_yolov3

def load_model_mmdet(device, name):
    from mmdet.apis import init_detector

    m = configs_mmdet[name]
    model_type, cfg_file, weight = m["model"], m["cfg"], m["weights"]
    
    cfg_options = dict(model=dict(type=model_type))
    model = init_detector(
        cfg_file,
        weight,
        device=device,
        cfg_options=cfg_options,
    )
    return model


if __name__ == '__main__':
    yolo3 = load_model_mmdet("cuda:7", "yolov3")
    print(yolo3)