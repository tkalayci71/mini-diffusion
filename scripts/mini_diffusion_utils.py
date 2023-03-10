#
# https://github.com/tkalayci71/mini-diffusion
#
# mini_diffusion_utils version 0.5
#

from modules import shared
import torch
import numpy as np
from scipy import integrate
from PIL import Image
from torchvision import transforms as tfms

#-------------------------------------------------------------------------------

def get_sd_version():
    result = 0
    frozen_embedder = shared.sd_model.cond_stage_model.wrapped
    if frozen_embedder.__class__.__name__=='FrozenCLIPEmbedder':
        result=1
    elif frozen_embedder.__class__.__name__=='FrozenOpenCLIPEmbedder':
        result=2
    return result

def get_tokenizer():
    result = None
    sdver = get_sd_version()
    if sdver==1:
        result = shared.sd_model.cond_stage_model.wrapped.tokenizer
    return result

def get_text_encoder():
    result = None
    sdver = get_sd_version()
    if sdver==1:
        result = shared.sd_model.cond_stage_model.wrapped.transformer
    return result

def get_unet():
    result = shared.sd_model.model.diffusion_model
    return result

def get_vae():
    result = shared.sd_model.first_stage_model
    return result

def get_model_info(model):
    torch_device = None
    data_type = None
    device_type = None
    if hasattr(model, 'parameters'):
        param = next(model.parameters(),None)
        if param!=None:
            torch_device = param.device
            device_type = str(torch_device).split(':', 1)[0]
            data_type = param.dtype
    return torch_device, data_type, device_type

#-------------------------------------------------------------------------------

def text_list_to_embeddings(text_list):
    tokenizer = get_tokenizer()
    text_encoder = get_text_encoder()
    torch_device, data_type, device_type = get_model_info(text_encoder)

    text_input = tokenizer(text_list, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    #text_input = tokenizer(text_list, padding="longest", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    input_ids = text_input.input_ids.to(torch_device)
    with torch.autocast(device_type=device_type, dtype=data_type):
        with torch.no_grad():
            text_embeddings = text_encoder(input_ids)[0]
    return text_embeddings

#-------------------------------------------------------------------------------

def generate_random_latents(seed,width,height,batch_size, torch_device, data_type, generator_device = None):
    if generator_device==None: generator_device=torch_device
    generator = torch.Generator(generator_device)
    generator.manual_seed(int(seed))
    latents = torch.randn((1, 4, height // 8, width // 8),generator=generator,device=generator_device)
    latents = torch.cat([latents]*batch_size)
    latents = latents.to(device=torch_device,dtype=data_type)
    return latents

#-------------------------------------------------------------------------------

def latents_to_pil_image(latents):
    vae = get_vae()
    torch_device, data_type, device_type = get_model_info(vae)
    latents = latents.to(device=torch_device, dtype=data_type)
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents)
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        result = [Image.fromarray(image) for image in images]
    return result

def pil_image_to_latent(image):
    vae = get_vae()
    torch_device, data_type, device_type = get_model_info(vae)
    with torch.no_grad():
        input_latent = tfms.ToTensor()(image).unsqueeze(0).to(device=torch_device,dtype=data_type)*2-1
        encoded = vae.encode(input_latent)
        result = 0.18215 * encoded.sample()
    return result

def tensor_image_to_latent(tensor):
    vae = get_vae()
    torch_device, data_type, device_type = get_model_info(vae)
    with torch.no_grad():
        input_latent = tensor.unsqueeze(0).to(device=torch_device,dtype=data_type)*2-1
        encoded = vae.encode(input_latent)
        result = 0.18215 * encoded.sample()
    return result

#-------------------------------------------------------------------------------

def prompts_to_images(prompts,negative_prompt,seed,cfg_scale,steps,width,height,randgen_device=None,image=None,denoising=1.0,brightness=1.0):

    sd_version = get_sd_version()
    if (sd_version!=1): return None, 'Error: SD1 only'

    batch_size = len(prompts)

    unet =get_unet()
    torch_device, data_type, device_type = get_model_info(unet)
    if randgen_device==None: randgen_device = torch_device

    num_train_timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.012
    betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    timesteps = torch.linspace(num_train_timesteps-1,0,steps,dtype=torch.int64)
    alphas_cumprod = alphas_cumprod[timesteps]
    image_weights = alphas_cumprod **0.5
    noise_weights = (1-alphas_cumprod) **0.5
    image_weights = torch.concat([image_weights,torch.tensor([1.0])])
    noise_weights = torch.concat([noise_weights,torch.tensor([0.0])])
    start_step = 0

    positive_embeddings = text_list_to_embeddings(prompts)
    negative_embeddings = text_list_to_embeddings([negative_prompt]*batch_size)
    text_embeddings = torch.cat([negative_embeddings,positive_embeddings])
    noise_latent = generate_random_latents(seed,width,height,batch_size,torch_device,data_type,generator_device=randgen_device)

    if image!=None:
        start_step = int(steps*(1.0-denoising))
        resized_image = image.resize((width,height))
        tensor_image = tfms.ToTensor()(resized_image) * brightness
        image_latent = tensor_image_to_latent(tensor_image)
        latents = noise_latent*noise_weights[start_step] + image_latent*image_weights[start_step]
    else:
        latents = noise_latent*noise_weights[start_step]

    print('--- mini diffusion ---')
    for step in range(len(timesteps)):
        if step<start_step:continue
        timestep = timesteps[step]
        print('step = ',step,'timestep=',timestep)
        latent_model_input = torch.cat([latents] * 2)
        tt = torch.tensor([timestep]*2*batch_size).to(device=torch_device,dtype=data_type)
        with torch.autocast(device_type=device_type, dtype=data_type):
            with torch.no_grad():
                noise_pred = unet.forward(latent_model_input, timesteps=tt, context = text_embeddings)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
        latents = (latents - noise_pred*noise_weights[step])/image_weights[step]
        latents = latents*image_weights[step+1] + noise_pred*noise_weights[step+1]

    result_images = latents_to_pil_image(latents)
    return result_images

#-------------------------------------------------------------------------------
