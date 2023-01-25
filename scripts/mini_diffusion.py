#
# https://github.com/tkalayci71/mini-diffusion
#
# mini_diffusion version 0.1
#

import torch
import scripts.mini_diffusion_utils as mdu
#import importlib
#importlib.reload(mdu)

#-------------------------------------------------------------------------------

def do_generate(prompt,negative_prompt,seed,cfg_scale,steps,width,height):

    sd_version = mdu.get_sd_version()
    if (sd_version!=1): return None

    unet =mdu.get_unet()
    torch_device, data_type, device_type = mdu.get_model_info(unet)
    scheduler = mdu.mini_lms()

    text_embeddings = mdu.text_list_to_embeddings([negative_prompt,prompt])
    latents = mdu.generate_random_latents(seed,width,height,1,torch_device,data_type,generator_device='cpu')
    scheduler.set_timesteps(steps, torch_device)
    latents = latents * scheduler.init_noise_sigma

    with torch.autocast(device_type=device_type, dtype=data_type):
        for i, t in enumerate(scheduler.timesteps):
            print('step = ',i)
            latent_model_input = torch.cat([latents] * 2)
            sigma = scheduler.sigmas[i]
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            tt = torch.tensor([t,t]).to(device=torch_device,dtype=data_type)
            with torch.no_grad():
                noise_pred = unet.forward(latent_model_input, timesteps=tt, context = text_embeddings)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
            latents = scheduler.step(noise_pred, t, latents)

    images = mdu.latents_to_pil_image(latents)
    result_image = images[0]

    return result_image

#-------------------------------------------------------------------------------

import gradio as gr
from modules.script_callbacks import on_ui_tabs

def add_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    prompt = gr.Textbox(label='Prompt',value='A watercolor painting of an otter')
                with gr.Row():
                    negative_prompt = gr.Textbox(label='Negative Prompt')
                with gr.Row():
                    seed = gr.Number(label='Seed',value=32)
                    cfg_scale = gr.Slider(label='CFG Scale',value=7.5,minimum=1,maximum=25,step=0.5)
                    steps = gr.Slider(label='Steps',value=30,minimum=1,maximum=50,step=1)
                with gr.Row():
                    width = gr.Slider(label='Width',value=512,minimum=256,maximum=768,step=64)
                    height = gr.Slider(label='Height',value=512,minimum=256,maximum=768,step=64)
                with gr.Row():
                    generate = gr.Button("Generate",variant='primary')
            with gr.Column(scale=1):
                result_image = gr.Image(show_label=False,interactive=False)

        generate.click(fn=do_generate,
            inputs=[prompt,negative_prompt,seed,cfg_scale,steps,width,height],
            outputs=[result_image])

    return [(ui, "Mini-Diffusion", "Mini_Diffusion")]

on_ui_tabs(add_tab)

#-------------------------------------------------------------------------------

