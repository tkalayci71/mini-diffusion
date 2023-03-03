#
# https://github.com/tkalayci71/mini-diffusion
#
# mini_diffusion version 0.5
#

import torch
import scripts.mini_diffusion_utils as mdu
import importlib
importlib.reload(mdu)

from time import perf_counter
from math import ceil
from PIL import Image
from io import BytesIO
import base64

def img_to_html(img,width,height):
    buffered = BytesIO()
    img.save(buffered,format="PNG")
    img_byte = buffered.getvalue()
    img_str = "data:image/png;base64," + base64.b64encode(img_byte).decode()
    result= '<img src="'+img_str+'" width='+str(width)+'height='+str(height)+'>'
    return result

#-------------------------------------------------------------------------------

def do_generate(prompt,negative_prompt,seed,cfg_scale,steps,width,height,rand_cpu,random_seed,image=None,denoising=1.0,brightness=1.0):

    clock_start = perf_counter()

    if random_seed==True:
        seed = int(torch.randint(high=0xffffffff,size=(1,)).item())
    prompts = prompt.split('\n')
    randgen_device='cpu' if rand_cpu else None
    result_images = mdu.prompts_to_images(prompts,negative_prompt,seed,cfg_scale,steps,width,height,randgen_device=randgen_device,image=image,denoising=denoising,brightness=brightness)

    clock_stop = perf_counter()
    result_text = 'time : '+str(ceil(clock_stop*1000-clock_start*1000))+' ms'

    html = '<table><tr>'
    for img in result_images:
        html += '<td>'+img_to_html(img,width,height)+'</td>'
    html += '</tr></table>'

    return html, result_text, seed

#-------------------------------------------------------------------------------

import gradio as gr
from modules.script_callbacks import on_ui_tabs

def add_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    prompt = gr.Textbox(label='Prompt',value='underwater, sunbeams, light rays, reflections, sparkle, shimmer, bubbles, seashells, pearls, glitter, bling',lines=2)
                with gr.Row():
                    negative_prompt = gr.Textbox(label='Negative Prompt',value='cartoon')
                with gr.Row():
                    steps = gr.Slider(label='Steps',value=10,minimum=0,maximum=100,step=1)
                    seed = gr.Number(label='Seed',value=0)
                    rand_cpu = gr.Checkbox(label='CPU',value=False)
                    random_seed = gr.Checkbox(label='Random',value=True)
                with gr.Row():
                    width = gr.Slider(label='Width',value=512,minimum=320,maximum=768,step=64)
                    height = gr.Slider(label='Height',value=512,minimum=320,maximum=768,step=64)
                    cfg_scale = gr.Slider(label='CFG Scale',value=7,minimum=1,maximum=25,step=0.5)
                with gr.Accordion(label='img2img',open=False):
                    image = gr.Image(type='pil')
                    denoising = gr.Slider(label='Denoising',value=0.7,minimum=0.0,maximum=1.0,step=0.01)
                    brightness = gr.Slider(label='Brightness',value=1.0,minimum=0.0,maximum=1.0,step=0.1)
                with gr.Row():
                    generate = gr.Button("Generate",variant='primary')
                with gr.Row():
                    result_text = gr.Textbox(show_label=False,interactive=False,lines=4)

            with gr.Column(scale=1):
                result_image = gr.HTML(show_label=False,interactive=False)

        generate.click(fn=do_generate,
            inputs=[prompt,negative_prompt,seed,cfg_scale,steps,width,height,rand_cpu,random_seed,image,denoising,brightness],
            outputs=[result_image,result_text,seed])

    return [(ui, "Mini-Diffusion", "Mini_Diffusion")]

on_ui_tabs(add_tab)

#-------------------------------------------------------------------------------

