import os
import torch
import argparse

from src.diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline


# reinstalling diffusers
os.system(os.getcwd() + "/src/diffusers/setup_sh.sh")

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--adv-coef', type=int, help='')
parser.add_argument('--type', type=str, help='standard|self|adv')
parser.add_argument('--steps', default=512, type=int, help='number of inference steps')
parser.add_argument('--device', default='cuda:0', help='')
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
prompt = "a hamburger on a table"

for p in pipe.unet.parameters():
    p.requires_grad = False

for p in pipe.text_encoder.parameters():
    p.requires_grad = False

for p in pipe.vae.parameters():
    p.requires_grad = False

if pargs.type == 'standard':
    print('Standard pipe')
    out = pipe(height=height, width=width, prompt=prompt, latents=latents,
              num_inference_steps=num_inference_steps)
    name = f'basic_{num_inference_steps}'
elif pargs.type == 'self':
    print('Self-guided pipe')
    self_guidance_dict = {"size": {"mode": "relative", "indices": [2], "values": [2.0]}}
    out = pipe(height=height, width=width, prompt=prompt, latents=latents, self_guidance_scale=15.0,
              num_inference_steps=num_inference_steps, self_guidance_dict=self_guidance_dict, )
    name = 'enlarged'
elif pargs.type == 'adv':
    print('Adv pipe')
    adv_guidance_scale = int(pargs.adv_coef)
    out = pipe(height=height, width=width, prompt=prompt, self_guidance_dict=dict(), latents=latents,
            num_inference_steps=num_inference_steps, self_guidance_scale=100.0, 
            adv_guidance_scale=adv_guidance_scale, adv_batch_size=20, adv_model='yolov2')
    name = f'adv_{num_inference_steps}_{adv_guidance_scale}'

out.images[0].show(title=name)
out.images[0].save(f'patches/{name}.png')
