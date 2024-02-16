import torch 
from torch.nn import Module
from torchvision.datasets.


@torch.no_grad()
def test_ap(
    model: Module,
    dataloader: DataLoader,
    renderer: RenderState,
    device: torch.device,
    args: configs.CLIConfig,
    tqdm_obj: Optional[tqdm] = None,
    num_samples=36,
):
    model.eval()
    criterion = DetectionLoss(
        args.iou_thresh_test,
        get_obj_cls_loss_fn(args.obj_cls_loss),
        reduction="mean_test",
        loss_type="max_iou",
    )
    preprocess = get_defense_preprocess(args).to(device)
    target_cls = get_target_cls(args)
    normalize = get_normalization(args).to(device)

    asr_meters = [
        utils.AttackSuccessRate(iou_threshold=args.iou_thresh_test)
        for _ in range(num_samples)
    ]
    map_meters = [
        MeanAveragePrecision(iou_thresholds=[args.iou_thresh_test])
        for _ in range(num_samples)
    ]
    test_time = SimpleMeter("test_time", "{:.3f}")
    test_loss_det = AverageMeter("test_loss_det", "{:.3f}")
    meters: list[Meter] = list(
        utils.filter_not_none(
            [
                test_time,
                test_loss_det,
            ]
        )
    )

    theta_list = np.linspace(-180, 180, num_samples, endpoint=False)
    # Light is always fixed to AmbientLights
    renderer.lights = renderer.light_sampler.sample(0)

    begin_time = time.time()
    for batch_idx, (data, _) in enumerate(dataloader):
        data: Tensor
        data = data.to(device)
        for angle_idx, theta in enumerate(theta_list):
            # Manually sample cameras
            renderer.cameras = renderer.camera_sampler.sample(len(data), theta=theta)
            if args.texture == "camouflage":
                tex_kwargs = dict(determinate=True)
            else:
                tex_kwargs = dict()
            render_kwargs = dict(use_tps2d=args.tps, use_tps3d=args.tps3d)
            patched, targets = renderer.forward(
                data,
                resample=False,
                is_test=True,
                share_texture=args.crop == "TCA",
                tex_kwargs=tex_kwargs,
                render_kwargs=render_kwargs,
            )
            patched, targets = apply_defense_preprocess(preprocess, patched, targets)
            all_boxes = get_boxes(model, normalize(patched), args)
            boxes = [box[idx == target_cls] for box, idx in all_boxes]
            det_loss = criterion.forward(boxes, targets)
            test_loss_det.update(det_loss.item(), len(data))
            for i, (boxes_with_score, cls_idx) in enumerate(all_boxes):
                boxes = boxes_with_score[:, :4]
                scores = boxes_with_score[:, 4]
                # Filter with NMS
                keep = nms(boxes, scores, args.nms_thresh)
                # Filter person class
                keep = keep[cls_idx[keep] == target_cls]
                boxes = boxes[keep]
                scores = scores[keep]
                cls_idx = cls_idx[keep]

                local_target = targets[i]
                target_labels = cls_idx.new_full((len(local_target),), target_cls)
                # The original metric requires absolute box coordinates,
                # but I don't think it would be different for relative ones
                preds = [dict(boxes=boxes, scores=scores, labels=cls_idx)]
                tgts = [dict(boxes=local_target, labels=target_labels)]
                asr_meters[angle_idx].update(preds, tgts)
                map_meters[angle_idx].update(preds, tgts)
            if tqdm_obj is not None:
                tqdm_obj.update()
    test_time.update(time.time() - begin_time)

    anglewise_meter_dict = dict(theta=theta_list, map=map_meters, asr=asr_meters)
    return meters, anglewise_meter_dict
