import os
import torch

from src.diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
# from src.diffusers.utils.self_guidance_utils import construct_guidance_dict

# reinstalling diffusers
os.system(os.getcwd() + "/src/diffusers/setup_sh.sh")

torch.manual_seed(0)
device = "cuda:0"

pipe = StableDiffusionPipeline.from_pretrained(os.getcwd() + "/src/diffusers/model", safety_checker=None).to(device)

batch_size = 1
num_channels_latents = pipe.unet.config.in_channels
height = pipe.unet.config.sample_size * pipe.vae_scale_factor
width = pipe.unet.config.sample_size * pipe.vae_scale_factor
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
num_inference_steps = 4

for p in pipe.unet.parameters():
    p.requires_grad = False

for p in pipe.text_encoder.parameters():
    p.requires_grad = False


print('Before pipe')
out = pipe(height=height, width=width, prompt=prompt, latents=latents,
           num_inference_steps=num_inference_steps)
out.images[0].show(title='basic')
out.images[0].save('basic.png')
print('After pipe')

self_guidance_dict = {"size": {"mode": "relative", "indices": [2], "values": [2.0]}}
out = pipe(height=height, width=width, prompt=prompt, self_guidance_dict=self_guidance_dict, latents=latents,
           num_inference_steps=num_inference_steps, self_guidance_scale=15.0)
out.images[0].show(title='enlarged')
out.images[0].save('enlarged.png')
