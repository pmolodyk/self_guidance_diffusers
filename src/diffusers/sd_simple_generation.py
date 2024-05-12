import os
import torch
import argparse
from torchvision import transforms as T
from PIL import Image
    
from src.diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--adv-coef', default='None', type=str, help='step_n_1:adv_coef_1 ... or adv_coef')
parser.add_argument('--type', type=str, help='standard|self|adv')
parser.add_argument('--prompt', default="colorful cat drawing", type=str, help='prompt for sd')
parser.add_argument('--adv-model', default="yolov3", type=str, help='yolov2|yolov3')
parser.add_argument('--steps', default=256, type=int, help='number of inference steps')
parser.add_argument('--device', default='cuda', help='')
parser.add_argument('--guidance-scale', default=7.5, type=float, help='guidance scale')
parser.add_argument('--save-every', default=-1, type=int, help='save result every save-every iterations')
parser.add_argument('--scale-type', default='basic', type=str, help='guidance scale coefficient type scheduler')
parser.add_argument('--input-image', default='', type=str, help='input image')
parser.add_argument('--fix-appearance', default=False, action="store_true", help='use self guidance for fixing appearance')
parser.add_argument('--fix-attention', default=False, action="store_true", help='use self guidance for fixing attention')
parser.add_argument('--self-guidance-scale', default=1000, type=float, help='self guidance scale')
parser.add_argument('--attention-weight', default=1, type=float, help='attention weight in self guidance loss')
parser.add_argument('--pipeline', default='3d', type=str, help='standard|3d')
parser.add_argument('--lo-steps', default=0, type=int, help='number of latent optimization steps')
parser.add_argument('--lo-coef', default=1e4, type=float, help='latent optimization scale')
parser.add_argument('--latents', default=None, type=str, help='latents_yolov2.pt')
parser.add_argument('--latents-search-n', default=1, type=int, help='Number of random latents to iterate over')
parser.add_argument('--latents-opt-mode', default=None, type=str, help='max|min|0...l-s-n')
parser.add_argument('--adv_batch_accum', default=-1, type=int, help='batch accumulation')
parser.add_argument('--name', default=None, type=str, help='run name for savefiles')


pargs = parser.parse_args()

device = pargs.device
num_inference_steps = int(pargs.steps)
torch.manual_seed(0)

model_name = 'model'
model_name = 'minisd'
pipe = StableDiffusionPipeline.from_pretrained(os.getcwd() + f"/src/diffusers/{model_name}", safety_checker=None).to(device)

batch_size = pargs.latents_search_n
num_channels_latents = pipe.unet.config.in_channels
if model_name != 'minisd':
    height = pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = pipe.unet.config.sample_size * pipe.vae_scale_factor
else:
    height = 256
    width = 256
dtp = pipe.text_encoder.dtype
default_generator = None

latents = None
latents_add = ""
if pargs.latents is not None:
    latents = torch.load(pargs.latents)
    latents_add = "_latents_" + pargs.latents
latents = pipe.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            dtp,
            device,
            default_generator,
            latents=latents,
        )

size_token_index = 2
prompt = pargs.prompt  # "colorful cat drawing"  # palm clothes pattern

for p in pipe.unet.parameters():
    p.requires_grad = False
 
for p in pipe.text_encoder.parameters():
    p.requires_grad = False

for p in pipe.vae.parameters():
    p.requires_grad = False

guidance_scale = int(pargs.guidance_scale)
self_guidance_scale = pargs.self_guidance_scale
save_every = pargs.save_every


print('Adv pipe')
n = num_inference_steps
adv_coef_split = pargs.adv_coef.split()
if len(adv_coef_split) == 1:
    adv_guidance_scale = int(pargs.adv_coef.split(':')[-1])
    adv_scale_schedule_dict = dict()
else:
    adv_scale_schedule_dict = dict([(int(s.split(':')[0]), int(s.split(':')[1])) for s in adv_coef_split])
    adv_guidance_scale = adv_scale_schedule_dict[0]
print(adv_guidance_scale, adv_scale_schedule_dict)
self_guidance_dict = {}
name = f'adv_{num_inference_steps}_{guidance_scale}_{pargs.adv_coef.replace(" ", "_")}'
if pargs.fix_appearance:
    self_guidance_dict['appearance'] = {'indices': list(range(1, len(prompt.split()) + 1))}
    name += f'_fx'
if pargs.fix_appearance or pargs.fix_attention:
    name += f'_{self_guidance_scale}'
if pargs.fix_attention:
    self_guidance_dict['fix_self_attention'] = {'weight': pargs.attention_weight}
    name += f'_fa_{pargs.attention_weight}'
if pargs.lo_steps > 0:
    name += f'_lo_{pargs.lo_steps}_{pargs.lo_coef}'
name += f'_{pargs.pipeline}'
if not pargs.adv_model.endswith('2'):
    name += f'_{pargs.adv_model}'
name += latents_add
if pargs.latents_opt_mode is not None:
    name += f'_los_{batch_size}_{pargs.latents_opt_mode}'
if pargs.adv_model in ('yolov2', 'yolov3', 'detr', 'yolov3-mmdet'):
    adv_bs = 12
elif pargs.adv_model == 'faster-rcnn':
    adv_bs = 9
else:
    adv_bs = 1

adv_accum = pargs.adv_batch_accum
name += f'_acc_{adv_accum}'

if pargs.name is not None:
    name = pargs.name

out = pipe(height=height, width=width, prompt=prompt, self_guidance_dict=self_guidance_dict, latents=latents,
           num_inference_steps=num_inference_steps, self_guidance_scale=self_guidance_scale,
           adv_guidance_scale=adv_guidance_scale, adv_batch_size=adv_bs, adv_accum_size=adv_accum,
           adv_model=pargs.adv_model, guidance_scale=guidance_scale, save_every=save_every,
           adv_scale_schedule_dict=adv_scale_schedule_dict, adv_scale_schedule_type=pargs.scale_type,
           self_guidance_precalculate_steps=num_inference_steps, pipeline=pargs.pipeline,
           num_latent_opt_steps=pargs.lo_steps, latent_opt_scale=pargs.lo_coef, latent_opt_mode=pargs.latents_opt_mode)

print('name', name)

out.images[0].show(title=name)
patch_path = f'patches/{"_".join(prompt.split())}'
patch_name = f'{name}_{prompt.replace(" ", "_")}.png'
os.makedirs(patch_path, exist_ok=True)
out.images[0].save(f'{patch_path}/{patch_name}')

