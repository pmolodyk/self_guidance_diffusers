import argparse
from .adversarial.utils.ap_calc_utils import get_save_aps


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--net', default='yolov2', help='target net name')
parser.add_argument('--batch-size', type=int, default=64, help='batch size')
parser.add_argument('--device', default='cuda:0', help='')
parser.add_argument('--load-path', default=None, help='load patch')
parser.add_argument('--mask', type=str, default=None, help='ex: adv_500*')
parser.add_argument('--no-save-res', default=False, action='store_true', help='')
pargs = parser.parse_args()

get_save_aps(pargs.device, pargs.load_path, pargs.mask, pargs.net, pargs.batch_size, pargs.no_save_res)
