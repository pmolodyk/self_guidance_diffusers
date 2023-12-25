import torch

def targets2padded(targets: torch.Tensor, max_len: int=15):  # (bs, 15, 5)
    device = targets[0].device
    results = [torch.concat((torch.zeros((1, 1)).to(device), target), axis=-1) for target in targets]
    res = [torch.concat((result, torch.full((max_len - targets[i].shape[0], targets[i].shape[1] + 1), -1).to(device)), axis=0).unsqueeze(0)
           for i, result in enumerate(results)]
    res = torch.concat(res, axis=0)
    return res
