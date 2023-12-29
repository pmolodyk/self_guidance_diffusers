import torch

def targets2padded(targets: torch.Tensor, max_len: int=15):  # (bs, 15, 5)
    res = torch.concat(targets)
    res[:, 0] = (res[:, 0] + res[:, 2]) / 2
    res[:, 1] = (res[:, 1] + res[:, 3]) / 2
    res[:, 2] = 2 * (res[:, 2] - res[:, 0])
    res[:, 3] = 2 * (res[:, 3] - res[:, 1])
    B, _ = res.shape
    device = res.device
    res = torch.concat((torch.zeros(B, 1).to(device), res), axis=1).unsqueeze(1)
    res = torch.concat((res, torch.full((B, max_len - 1, res.shape[-1]), -1).to(device)), axis=1)
    return res
