#
# https://github.com/tkalayci71/mini-diffusion
#
# mini_diffusion_utils version 0.2
#

from modules import shared
import torch
import numpy as np
from scipy import integrate
from PIL import Image
from time import perf_counter
from math import ceil

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

#-------------------------------------------------------------------------------

# simplifed version of diffusers/schedulers/scheduling_lms_discrete.py
class mini_lms:
    def __init__(self):
        self.num_train_timesteps: int = 1000
        self.beta_start: float = 0.00085
        self.beta_end: float = 0.012
        self.beta_schedule: str = "scaled_linear"
        self.betas = (torch.linspace(self.beta_start**0.5, self.beta_end**0.5, self.num_train_timesteps, dtype=torch.float32) ** 2)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = np.concatenate([sigmas[::-1], [0.0]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas)
        self.init_noise_sigma = self.sigmas.max()

    def set_timesteps(self, num_inference_steps: int, device):
        self.num_inference_steps = num_inference_steps
        timesteps = np.linspace(0, self.num_train_timesteps - 1, num_inference_steps, dtype=float)[::-1].copy()
        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas).to(device=device)
        self.timesteps = torch.from_numpy(timesteps).to(device=device)
        self.derivatives = []

    def scale_model_input(self, sample: torch.FloatTensor, timestep):
        timestep = timestep.to(self.timesteps.device)
        step_index = (self.timesteps == timestep).nonzero().item()
        sigma = self.sigmas[step_index]
        divval = ((sigma**2 + 1) ** 0.5)
        sample = sample / divval
        return sample

    def get_lms_coefficient(self, order, t, current_order):
        def lms_derivative(tau):
            prod = 1.0
            for k in range(order):
                if current_order == k:
                    continue
                prod *= (tau - self.sigmas[t - k]) / (self.sigmas[t - current_order] - self.sigmas[t - k])
            return prod
        integrated_coeff = integrate.quad(lms_derivative, self.sigmas[t], self.sigmas[t + 1], epsrel=1e-4)[0]
        return integrated_coeff

    def step(self, model_output: torch.FloatTensor, step_index, sample: torch.FloatTensor,order: int = 4) :
        sigma = self.sigmas[step_index]
        pred_original_sample = sample - sigma * model_output
        derivative = (sample - pred_original_sample) / sigma
        self.derivatives.append(derivative)
        if len(self.derivatives) > order: self.derivatives.pop(0)
        order = min(step_index + 1, order)
        lms_coeffs = [self.get_lms_coefficient(order, step_index, curr_order) for curr_order in range(order)]
        prev_sample = sample + sum(coeff * derivative for coeff, derivative in zip(lms_coeffs, reversed(self.derivatives)))
        return prev_sample

#-------------------------------------------------------------------------------

def prompts_to_images(prompts,negative_prompt,seed,cfg_scale,steps,width,height,randgen_device=None):

    sd_version = get_sd_version()
    if (sd_version!=1): return None, 'Error: SD1 only'

    batch_size = len(prompts)

    log = []

    total_clock_start = perf_counter()

    clock_start = perf_counter()
    unet =get_unet()
    torch_device, data_type, device_type = get_model_info(unet)
    if randgen_device==None:
        randgen_device = torch_device

    scheduler = mini_lms()
    positive_embeddings = text_list_to_embeddings(prompts)
    negative_embeddings = text_list_to_embeddings([negative_prompt]*batch_size)
    text_embeddings = torch.cat([negative_embeddings,positive_embeddings])
    latents = generate_random_latents(seed,width,height,batch_size,torch_device,data_type,generator_device=randgen_device)
    scheduler.set_timesteps(steps,'cpu')
    latents = latents * scheduler.init_noise_sigma
    clock_stop = perf_counter()
    log.append('prep time : '+str(ceil(clock_stop*1000-clock_start*1000))+' ms')

    clock_start = perf_counter()
    with torch.autocast(device_type=device_type, dtype=data_type):
        for i, t in enumerate(scheduler.timesteps):
            #print('step = ',i)
            latent_model_input = torch.cat([latents] * 2)
            sigma = scheduler.sigmas[i]
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            tt = torch.tensor([t]*2*batch_size).to(device=torch_device,dtype=data_type)
            with torch.no_grad():
                noise_pred = unet.forward(latent_model_input, timesteps=tt, context = text_embeddings)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
            latents = scheduler.step(noise_pred, i, latents)
    clock_stop = perf_counter()
    log.append('diffusion time : '+str(ceil(clock_stop*1000-clock_start*1000))+' ms')

    clock_start = perf_counter()
    result_images = latents_to_pil_image(latents)
    clock_stop = perf_counter()
    log.append('vae decode time: '+str(ceil(clock_stop*1000-clock_start*1000))+' ms')

    total_clock_stop = perf_counter()
    log.append('total time : '+str(ceil(total_clock_stop*1000-total_clock_start*1000))+' ms')

    result_text = '\n'.join(log)
    return result_images, result_text

#-------------------------------------------------------------------------------

