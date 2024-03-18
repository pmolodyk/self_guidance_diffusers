import yaml
import os
from tqdm import tqdm 
from time import sleep
import argparse
from src.diffusers.adversarial.utils.server_utils import check_free
from src.diffusers.adversarial.utils.ap_calc_utils import get_save_aps

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--yaml-name', type=str)
pargs = parser.parse_args()

with open(f'src/diffusers/{pargs.yaml_name}.yaml') as f:
    data_dict = yaml.load(f, Loader=yaml.SafeLoader)

n = len(data_dict['adv_scale_schedule_dict'])

device = data_dict['device']
prompt = data_dict['prompt']
gsc = data_dict['guidance_scale']
fap = data_dict['fix_appearance']
fat = data_dict['fix_attention']
adv_model = data_dict['adv_model']
lo_steps = data_dict['lo_steps'] if 'lo_steps' in data_dict else [0] * n
lo_coef = data_dict['lo_coef'] if int(lo_steps[0]) > 0 else [0] * n

if data_dict['check_free']:
    check_free([data_dict['server_name']], [int(device[-1])])

for i in tqdm(range(n), total=n):
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
    lo_text = ""
    if lo_steps[i] > 0:
        lo_text = f"--lo-steps {lo_steps[i]} --lo-coef {lo_coef[i]}"

    cmd = f'python -m src.diffusers.sd_simple_generation --adv-coef "{dct}" --adv-model "{adv_model}" --type adv --steps {steps} --device {device} --prompt "{prompt}" --guidance-scale {gsc} {fa_text} {lo_text}'
    print(cmd)
    os.system(cmd)

    if 'ap_models' in data_dict:
        sleep(5)
        name = f'adv_{steps}_{int(gsc)}_{dct.replace(" ", "_")}'
        if len(fa_text) > 1:
            name += f'_fx'
        if fat or fap:
            name += f"_{float(data_dict['appearance_coef'][i])}"
        if fat:
            name += f'_fa_{att_weight}'
        if lo_steps[i] > 0:
            name += f'_lo_{lo_steps[i]}_{lo_coef[i]}'
        name += '_3d'
        if not adv_model.endswith('2'):
            name += f'_{adv_model}'
        patch_path = f'patches/{"_".join(prompt.split())}'
        patch_name = f'{name}_{prompt.replace(" ", "_")}.png'
        print('patch_path:', patch_path)
        print('patch_name:', patch_name)
        for now_model in data_dict["ap_models"]:
            cmd = f'python -m src.diffusers.test_3d --load-path "{patch_path}" --mask "{patch_name}" --device {device} --net "{now_model}"'
            print(cmd)
            os.system(cmd)
            sleep(5)

