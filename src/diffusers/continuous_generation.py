import yaml
import os
from tqdm import tqdm 
import argparse
from src.diffusers.adversarial.utils.server_utils import check_free
    
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--yaml-name', type=str)
pargs = parser.parse_args()

with open(f'src/diffusers/{pargs.yaml_name}.yaml') as f:
    data_dict = yaml.load(f, Loader=yaml.SafeLoader)

device = data_dict['device']
prompt = data_dict['prompt']
gsc = data_dict['guidance_scale']
fap = data_dict['fix_appearance']
fat = data_dict['fix_attention']

if data_dict['check_free']:
    check_free([data_dict['server_name']], [int(device[-1])])

for i in tqdm(range(len(data_dict["adv_scale_schedule_dict"])), total=len(data_dict["adv_scale_schedule_dict"])):
    adv_scale_schedule_dict = data_dict['adv_scale_schedule_dict'][i]
    adv_scale_schedule_dict_new = {}
    steps = int(data_dict["n"])
    for (k, v) in adv_scale_schedule_dict.items():
        if float(k) != int(k):
            k = int(float(k) * steps)
        adv_scale_schedule_dict_new[k] = v
    dct = str(adv_scale_schedule_dict_new)[1:-1].replace(': ', ':').replace(',', '')
    fa_text = ''
    if 'attention_weight' not in data_dict or len(data_dict['attention_weight']) == 0:
        att_weight = 1
    else:
        att_weight = data_dict['attention_weight'][i]
    if fap:
        fa_text += f"--fix-appearance "
    if fat:
        fa_text += f"--fix-attention --attention-weight {att_weight} "
    if fat or fap:
        fa_text += f"--self-guidance-scale {data_dict['appearance_coef'][i]}"

    cmd = f'python -m src.diffusers.sd_simple_generation --adv-coef "{dct}" --type adv --steps {steps} --device {device} --prompt "{prompt}" --guidance-scale {gsc} {fa_text}'
    print(cmd)
    os.system(cmd)
