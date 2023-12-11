import os
import torch
import argparse
    
from src.diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline


# reinstalling diffusers
os.system(os.getcwd() + "/src/diffusers/setup_sh.sh")

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--adv-coef', type=str, help='step_n_1:adv_coef_1 ... or adv_coef')
parser.add_argument('--type', type=str, help='standard|self|adv')
parser.add_argument('--prompt', default="colorful cat drawing", type=str, help='prompt for sd')
parser.add_argument('--steps', default=125, type=int, help='number of inference steps')
parser.add_argument('--device', default='cuda:0', help='')
parser.add_argument('--guidance-scale', default=7.5, type=float, help='guidance scale')
parser.add_argument('--save-every', default=-1, type=int, help='save result every save-every iterations')
parser.add_argument('--scale-type', default='basic', type=str, help='guidance scale coefficient type scheduler')

pargs = parser.parse_args()

device = pargs.device
assert pargs.type in ('standard', 'self', 'adv')
num_inference_steps = int(pargs.steps)
torch.manual_seed(0)

pipe = StableDiffusionPipeline.from_pretrained(os.getcwd() + "/src/diffusers/minisd", safety_checker=None).to(device)

batch_size = 1
num_channels_latents = pipe.unet.config.in_channels
height = 256  # pipe.unet.config.sample_size * pipe.vae_scale_factor
width = 256  # pipe.unet.config.sample_size * pipe.vae_scale_factor
dtp = pipe.text_encoder.dtype
default_generator = None

latents = pipe.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            dtp,
            device,
            default_generator,
            None,
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
save_every = int(pargs.save_every)
if pargs.type == 'standard':
    print('Standard pipe')
    out = pipe(height=height, width=width, prompt=prompt, latents=latents,
               num_inference_steps=num_inference_steps, save_every=save_every)
    name = f'basic_{num_inference_steps}'
elif pargs.type == 'self':
    print('Self-guided pipe')
    self_guidance_dict = {"size": {"mode": "relative", "indices": [5], "values": [0.01]}}
    self_guidance_dict = {"position": {"mode": "relative", "indices": [5], "values": [0.01]}}
    out = pipe(height=height, width=width, prompt=prompt, latents=latents, self_guidance_scale=15., 
               num_inference_steps=num_inference_steps, self_guidance_dict=self_guidance_dict, save_every=save_every,
               self_guidance_precalculate_steps=num_inference_steps)
    name = 'enlarged'
elif pargs.type == 'adv':
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
    out = pipe(height=height, width=width, prompt=prompt, self_guidance_dict=dict(), latents=latents,
            num_inference_steps=num_inference_steps, self_guidance_scale=100.0, 
            adv_guidance_scale=adv_guidance_scale, adv_batch_size=24, adv_model='yolov2',
            guidance_scale=guidance_scale, save_every=save_every, adv_scale_schedule_dict=adv_scale_schedule_dict,
            adv_scale_schedule_type=pargs.scale_type)
    name = f'adv_{num_inference_steps}_{guidance_scale}_{pargs.adv_coef.replace(" ", "_")}'

out.images[0].show(title=name)
out.images[0].save(f'patches/{name}_{prompt.replace(" ", "_")}.png')
