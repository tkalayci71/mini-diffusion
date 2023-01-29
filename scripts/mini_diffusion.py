#
# https://github.com/tkalayci71/mini-diffusion
#
# mini_diffusion version 0.2
#

import torch
import scripts.mini_diffusion_utils as mdu
import importlib
importlib.reload(mdu)

#-------------------------------------------------------------------------------

def do_generate(prompt,negative_prompt,seed,cfg_scale,steps,width,height,rand_cpu):
    prompts = prompt.split('\n')
    randgen_device='cpu' if rand_cpu else None
    result_images, result_text = mdu.prompts_to_images(prompts,negative_prompt,seed,cfg_scale,steps,width,height,randgen_device=randgen_device)
    return result_images, result_text

#-------------------------------------------------------------------------------

import gradio as gr
from modules.script_callbacks import on_ui_tabs

def add_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    prompt = gr.Textbox(label='Prompt',value='A watercolor painting of an otter',lines=2)
                with gr.Row():
                    negative_prompt = gr.Textbox(label='Negative Prompt')
                with gr.Row():
                    with gr.Column():
                        seed = gr.Number(label='Seed',value=32)
                        rand_cpu = gr.Checkbox(label='CPU random generator',value=True)
                    with gr.Column():
                        cfg_scale = gr.Slider(label='CFG Scale',value=7.5,minimum=0,maximum=15,step=0.5)
                        steps = gr.Slider(label='Steps',value=30,minimum=0,maximum=50,step=1)
                with gr.Row():
                    width = gr.Slider(label='Width',value=512,minimum=256,maximum=768,step=64)
                    height = gr.Slider(label='Height',value=512,minimum=256,maximum=768,step=64)
                with gr.Row():
                    generate = gr.Button("Generate",variant='primary')
                with gr.Row():
                    result_text = gr.Textbox(show_label=False,interactive=False,lines=2,maxlines=2)

            with gr.Column(scale=1):
                result_image = gr.Gallery(show_label=False,interactive=False)

        generate.click(fn=do_generate,
            inputs=[prompt,negative_prompt,seed,cfg_scale,steps,width,height,rand_cpu],
            outputs=[result_image,result_text])

    return [(ui, "Mini-Diffusion", "Mini_Diffusion")]

on_ui_tabs(add_tab)

#-------------------------------------------------------------------------------

