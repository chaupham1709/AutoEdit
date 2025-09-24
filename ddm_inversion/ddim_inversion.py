
from ddm_inversion.inversion_utils import encode_text
from typing import Union
import torch
import numpy as np
from tqdm import tqdm

def next_step(model, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
    timestep, next_timestep = min(timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep] if timestep >= 0 else model.scheduler.final_alpha_cumprod
    alpha_prod_t_next = model.scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample

def get_noise_pred(model, latent, t, context, cfg_scale):
    latents_input = torch.cat([latent] * 2)
    noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + cfg_scale * (noise_prediction_text - noise_pred_uncond)
    # latents = next_step(model, noise_pred, t, latent)
    return noise_pred

@torch.no_grad()
def ddim_loop(model, w0, prompt, cfg_scale):
    # uncond_embeddings, cond_embeddings = self.context.chunk(2)
    # all_latent = [latent]
    text_embedding = encode_text(model, prompt)
    uncond_embedding = encode_text(model, "")
    context = torch.cat([uncond_embedding, text_embedding])
    latent = w0.clone().detach()
    latent_list = [latent]
    noise_list = []
    for i in tqdm(range(model.scheduler.num_inference_steps)):
        t = model.scheduler.timesteps[len(model.scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred(model, latent, t, context, cfg_scale)
        latent = next_step(model, noise_pred, t, latent)
        latent_list.append(latent)
        noise_list.append(noise_pred)
        # all_latent.append(latent)
    return latent, noise_list, latent_list

@torch.no_grad()
def ddim_inversion(model, w0, prompt, cfg_scale):
    wT = ddim_loop(model, w0, prompt, cfg_scale)
    return wT

@torch.no_grad()
def reconstruct_ddim_inversion(model, wT, prompt, cfg_scale, latent_list, noise_list):
    text_embedding = encode_text(model, prompt)
    uncond_embedding = encode_text(model, "")
    context = torch.cat([uncond_embedding, text_embedding])
    latent = wT.clone().detach()
    i = len(latent_list) -1
    j = len(noise_list) - 1
    for t in tqdm(model.scheduler.timesteps):
        noise_pred = get_noise_pred(model, latent, t, context, cfg_scale)
        print(f'Step {i}: noise_pred {torch.mean(torch.abs(noise_list[j] - noise_pred))}, latent {torch.mean(torch.abs(latent_list[i] - latent))}')
        latent = model.scheduler.step(noise_pred, t, latent)["prev_sample"]
        i-=1
        j-=1
    
    return latent