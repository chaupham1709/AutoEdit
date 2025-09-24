from torch.nn import functional as F
import torch
from prompt_to_prompt import ptp_utils
from typing import List, Union, Optional, Tuple, Dict
import numpy as np
from PIL import Image
import abc
from prompt_to_prompt import seq_aligner
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torch.optim.adam import Adam
from pathlib import Path
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss
from actor_critic_null_text import UnetDownSampling, UnetActorDiscrete, UnetCriticDiscrete
# from datasets import EditingDataset, collate_fn
import argparse
import os
from loss import PolicyLoss, ValueLoss
from reward import RewardFunction
from dataclasses import dataclass

LOW_RESOURCE = False 
NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77

@dataclass
class Experience:
    latents: torch.Tensor
    action_list: torch.Tensor
    hidden_feature_list: torch.Tensor
    log_probs: torch.Tensor
    kl_penalized_reward: torch.Tensor
    advantage: torch.Tensor
    estimate_kl: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor

device = "cuda"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--work_dir', type=str, default='exp')
    parser.add_argument('--train_first_stage', action='store_true', default=False)
    parser.add_argument('--annotation_folder', type=str, default="Pie_Bench/")
    parser.add_argument("--annotation_file", type=str, default="all_train_val.json")
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--report_to", type=str, default="tensorboard",)
    parser.add_argument("--checkpointing_steps", type=int, default=1)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--stop_timestep", type=int, default=20)
    parser.add_argument("--input_image_path", type=str)
    parser.add_argument("--src_prompt", type=str)
    parser.add_argument("--tgt_prompt", type=str)
    parser.add_argument("--save_edit_path", type=str)

    args = parser.parse_args()
    return args

class LocalBlend:
    
    def get_mask(self, maps, alpha, use_pool):
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = F.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        
        mask = F.interpolate(maps, size=(64,64))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1-int(use_pool)])
        mask = mask[:1] + mask
        return mask
    
    def __call__(self, x_t, attention_store):
        self.counter += 1
        if self.counter > self.start_blend:
           
            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
            maps = torch.cat(maps, dim=1)
            mask = self.get_mask(maps, self.alpha_layers, True)
            if self.substruct_layers is not None:
                maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
                mask = mask * maps_sub
            mask = mask.float()
            x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], tokenizer, substruct_words=None, start_blend=0.2, th=(.3, .3)):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        
        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * NUM_DDIM_STEPS)
        self.counter = 0 
        self.th=th
    
class EmptyControl:
    
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn

class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class SpatialReplace(EmptyControl):
    
    def step_callback(self, x_t):
        if self.cur_step < self.stop_inject:
            b = x_t.shape[0]
            x_t = x_t[:1].expand(b, *x_t.shape[1:])
        return x_t

    def __init__(self, stop_inject: float):
        super(SpatialReplace, self).__init__()
        self.stop_inject = int((1 - stop_inject) * NUM_DDIM_STEPS)
        

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

        
class AttentionControlEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] <= 32 ** 2:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend], tokenizer):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, tokenizer,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend, tokenizer)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)
        

class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, tokenizer,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend, tokenizer)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer, tokenizer,
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend, tokenizer)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller

def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image

class NullTextInversion:
    def __init__(self,
                 pipeline):
        self.model = pipeline
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
    
    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        context = torch.cat([uncond_embeddings, text_embeddings])
        return context
    
    def invert(self, image_path: str, prompt: str, offsets=(0,0,0,0), num_inner_steps=10, early_stop_epsilon=1e-5):
        context = self.init_prompt(prompt)
        # ptp_utils.register_attention_control(self.model, None)
        image_gt = load_512(image_path, *offsets)
        image_rec, ddim_latents = self.ddim_inversion(image_gt, context)
        uncond_embeddings = self.null_optimization(context, ddim_latents, num_inner_steps, early_stop_epsilon)
        return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings
    
    @torch.no_grad()
    def ddim_loop(self, latent, context):
        uncond_embeddings, cond_embeddings = context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(NUM_DDIM_STEPS):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent
    
    @torch.no_grad()
    def ddim_inversion(self, image, context):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent, context)
        return image_rec, ddim_latents
    
    def null_optimization(self, context, latents, num_inner_steps, epsilon):
        uncond_embeddings, cond_embeddings = context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * NUM_DDIM_STEPS)
        for i in range(NUM_DDIM_STEPS):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = F.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
        bar.close()
        return uncond_embeddings_list
    
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else GUIDANCE_SCALE
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents
    
    @property
    def scheduler(self):
        return self.model.scheduler
    
    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents
    
    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]], tokenizer):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)
    
    for word, val in zip(word_select, values):
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer

def make_controller(prompts: List[str], tokenizer, is_replace_controller: bool, cross_replace_steps: Dict[str, float], self_replace_steps: float, blend_words=None, equilizer_params=None) -> AttentionControlEdit:
    if blend_words is None:
        lb = None
    else:
        lb = LocalBlend(prompts, blend_words, tokenizer)
    # lb = None
    if is_replace_controller:
        controller = AttentionReplace(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, tokenizer=tokenizer, local_blend=lb)
    else:
        controller = AttentionRefine(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, tokenizer=tokenizer, local_blend=lb)
    if equilizer_params is not None:
        eq = get_equalizer(prompts[1], equilizer_params["words"], equilizer_params["values"], tokenizer)
        controller = AttentionReweight(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps,
                                       self_replace_steps=self_replace_steps, equalizer=eq, local_blend=lb, controller=controller, tokenizer=tokenizer)
    return controller

@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    skip_timestep=15,
    start_time=50,
    return_type='image'
):
    batch_size = len(prompt) // 2
    ptp_utils.register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        if uncond_embeddings_ is None:
            if i < skip_timestep:
                cond_embeddings = text_embeddings[:1]
            else:
                cond_embeddings = text_embeddings[1:]
            context = torch.cat([uncond_embeddings[i].expand(*cond_embeddings.shape), cond_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = ptp_utils.diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False)
        
    if return_type == 'image':
        image = ptp_utils.latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent

class NullTextInversionWithStableDiffusion:
    def __init__(self, dataset, actor, args, sft_model, critic):
        self.dataset = dataset
        self.actor = actor
        self.criterion = CrossEntropyLoss()
        self.args = args
        self.num_inference_steps = NUM_DDIM_STEPS
        self.sft_model = sft_model
        self.critic = critic
        self.policy_loss = PolicyLoss()
        self.value_loss = ValueLoss()
        self.reward_fn = RewardFunction(3.0, 1.0, "cpu")
        self.actor_lr = 5e-5
        self.critic_lr = 5e-5
        self.kl_beta = 0.02
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
    
    def init_model(self):
        self.pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        self.pipeline.scheduler = DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
        self.null_inversion = NullTextInversion(self.pipeline)
        self.pipeline.scheduler.set_timesteps(NUM_DDIM_STEPS)

        unet = self.pipeline.unet
        del unet
        unet = UnetDownSampling.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
        self.pipeline.unet = unet

        self.pipeline.to("cuda")
        self.actor.to("cuda")
    
    @torch.no_grad()
    def decode_latents(self, latents):
        return ptp_utils.latent2image(self.pipeline.vae, latents)
    
    @torch.no_grad()
    def ddim_inversion(self, image_path, prompt):
        context = self.null_inversion.init_prompt(prompt)
        offsets = (0, 0, 0, 0)
        image = load_512(image_path, *offsets)
        _, latents = self.null_inversion.ddim_inversion(image, context)
        return latents[-1]

    
    def inversion_process(self, image_path, prompt):
        _, latents, uncond_embeddings = self.null_inversion.invert(image_path, prompt, offsets=(0,0,0,0))
        return latents, uncond_embeddings
    
    @torch.no_grad()
    def denoising_trajectory(self, latents, denoising_trajectory, src_prompt, tgt_prompt, uncond_embeddings, verbose=False):
        height = width = 512
        prompt = [src_prompt, tgt_prompt]
        batch_size = latents.shape[0]
    
        text_input = self.pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.pipeline.text_encoder(text_input.input_ids.to(self.pipeline.device))[0]
        max_length = text_input.input_ids.shape[-1]
        latent, latents = ptp_utils.init_latent(latents, self.pipeline, height, width, None, batch_size)
        
        hidden_feature_list = []
        if verbose:
            timesteps = tqdm(self.pipeline.scheduler.timesteps)
        else:
            timesteps = self.pipeline.scheduler.timesteps
        for i, t in enumerate(timesteps):
            src_embedding = text_embeddings[:1]
            tgt_embedding = text_embeddings[1:]
            hidden_feature_src, src_noise_pred = self.pipeline.unet(
                latents,
                t,
                encoder_hidden_states=src_embedding,
                return_dict=False,
                return_hidden = True
            )
            hidden_feature_tgt, tgt_noise_pred = self.pipeline.unet(
                latents,
                t,
                encoder_hidden_states=tgt_embedding,
                return_dict=False,
                return_hidden = True
            )
            hidden_feature_list.append(torch.cat([hidden_feature_src, hidden_feature_tgt], dim=1))
            uncond_embedding_ = uncond_embeddings[i].expand(*src_embedding.shape).to(src_embedding.device)
            uncond_noise_pred = self.pipeline.unet(
                latents,
                t,
                encoder_hidden_states=uncond_embedding_,
                return_dict=False,
            )[0]
            src_noise_pred = uncond_noise_pred + GUIDANCE_SCALE * (src_noise_pred - uncond_noise_pred)
            tgt_noise_pred = uncond_noise_pred + GUIDANCE_SCALE * (tgt_noise_pred - uncond_noise_pred)
            masks = (denoising_trajectory[:, i]==1).float()
            noise_pred = tgt_noise_pred * masks[:, None, None, None] + (1-masks[:, None, None, None]) * src_noise_pred
            latents = self.pipeline.scheduler.step(
                noise_pred, t, latents
            ).prev_sample
        
        hidden_feature_list = torch.stack(hidden_feature_list, dim=1)
        return latents, hidden_feature_list
    
    def freeze_parameters(self):
        self.pipeline.unet.requires_grad_(False)
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)
    
    def make_random_trajectory(self, batch_size, device):
        denoising_trajectories = torch.ones((batch_size, self.num_inference_steps),device=device).long()
        inject_steps = torch.randint(12, 30, (batch_size,))
        for i, step in enumerate(inject_steps):
            denoising_trajectories[i, :step] = 0
        
        return denoising_trajectories
    
    @torch.no_grad()
    def batch_generate(self, latents, src_prompt, tgt_prompt, uncond_embeddings, verbose=False):
        height = width = 512
        prompt = [src_prompt, tgt_prompt]
        batch_size = latents.shape[0]
    
        text_input = self.pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.pipeline.text_encoder(text_input.input_ids.to(self.pipeline.device))[0]
        max_length = text_input.input_ids.shape[-1]
        latent, latents = ptp_utils.init_latent(latents, self.pipeline, height, width, None, batch_size)
        
        hidden_feature_list = []
        action_list = []
        if verbose:
            timesteps = tqdm(self.pipeline.scheduler.timesteps)
        else:
            timesteps = self.pipeline.scheduler.timesteps
        
        for i, t in enumerate(timesteps):
            src_embedding = text_embeddings[:1]
            tgt_embedding = text_embeddings[1:]
            hidden_feature_src, src_noise_pred = self.pipeline.unet(
                latents,
                t,
                encoder_hidden_states=src_embedding,
                return_dict=False,
                return_hidden = True
            )
            hidden_feature_tgt, tgt_noise_pred = self.pipeline.unet(
                latents,
                t,
                encoder_hidden_states=tgt_embedding,
                return_dict=False,
                return_hidden = True
            )
            hidden_features = torch.cat([hidden_feature_src, hidden_feature_tgt], dim=1)
            timestep = torch.cat([t.unsqueeze(0)] * batch_size).to(latent.device)
            logits = self.actor(hidden_features, timestep)
            probs = F.softmax(logits, dim=1)
            next_action = torch.multinomial(probs, num_samples=1)
            action_list.append(next_action)
            hidden_feature_list.append(torch.cat([hidden_feature_src, hidden_feature_tgt], dim=1))
            uncond_embedding_ = uncond_embeddings[i].expand(*src_embedding.shape).to(src_embedding.device)
            uncond_noise_pred = self.pipeline.unet(
                latents,
                t,
                encoder_hidden_states=uncond_embedding_,
                return_dict=False,
            )[0]
            
            src_noise_pred = uncond_noise_pred + GUIDANCE_SCALE * (src_noise_pred - uncond_noise_pred)
            tgt_noise_pred = uncond_noise_pred + GUIDANCE_SCALE * (tgt_noise_pred - uncond_noise_pred)
            masks = (next_action[:, 0]==1).float()
            noise_pred = tgt_noise_pred * masks[:, None, None, None] + (1-masks[:, None, None, None]) * src_noise_pred
            latents = self.pipeline.scheduler.step(
                noise_pred, t, latents
            ).prev_sample
        
        hidden_feature_list = torch.stack(hidden_feature_list, dim=1)
        action_list = torch.cat(action_list, dim=1)
        return latents, hidden_feature_list, action_list
    
    @torch.no_grad()
    def make_experience(self, image_path, latents, src_prompt, tgt_prompt, uncond_embeddings, masks, accelerator):
        self.sft_model.eval()
        self.actor.eval()
        self.critic.eval()
        latents, hidden_feature_list, action_list = self.batch_generate(
            latents, src_prompt, tgt_prompt, uncond_embeddings, verbose=False
        )
        batch_size = latents.shape[0]
        timesteps = self.pipeline.scheduler.timesteps.to(latents.device)
        timesteps = torch.cat([timesteps] * batch_size).reshape([-1])

        edit_images = self.decode_latents(latents)
        original_images = np.array(Image.open(image_path).convert("RGB"))[None,:,:,:]
        editing_region_masks = masks.permute(0,2,3,1).cpu().numpy()
        reward = self.reward_fn.estimate_reward(
            original_images,
            edit_images,
            [tgt_prompt],
            editing_region_masks
        )
        reward = reward.to(latents.device).unsqueeze(-1)

        if accelerator is not None:
            unwrapped_actor = accelerator.unwrap_model(self.actor)
            unwrapped_sft_model = accelerator.unwrap_model(self.sft_model)
            unwrapped_critic = accelerator.unwrap_model(self.critic)
        
        log_probs = unwrapped_actor.forward_actor(
            hidden_feature_list, timesteps, action_list
        )
        sft_log_probs = unwrapped_sft_model.forward_actor(
            hidden_feature_list, timesteps, action_list
        )
        values = unwrapped_critic.forward_critic(
            hidden_feature_list, timesteps
        )
        log_probs = log_probs.squeeze(-1)
        sft_log_probs = sft_log_probs.squeeze(-1)
        kl_penalized_reward, estimate_kl = \
            self.kl_penalized_reward(
                reward,
                log_probs,
                sft_log_probs
            )
        
        advantage = kl_penalized_reward - values

        return Experience(
            latents,
            action_list,
            hidden_feature_list,
            log_probs,
            kl_penalized_reward,
            advantage,
            estimate_kl,
            values,
            reward
        )    
    
    def train_first_stage(self):
        logging_dir = Path(self.args.work_dir, "logging")
        accelerator_project_config = ProjectConfiguration(project_dir=self.args.work_dir, logging_dir=logging_dir)

        accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=self.args.mixed_precision,
            log_with=self.args.report_to,
            project_config=accelerator_project_config,
        )
        if accelerator.is_main_process:
            if self.args.work_dir is not None:
                os.makedirs(self.args.work_dir, exist_ok=True)
                os.makedirs(os.path.join(self.args.work_dir, "logging"), exist_ok=True)
            writer = SummaryWriter(log_dir=os.path.join(self.args.work_dir, "logging"))
        
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
        )
        unet = self.pipeline.unet
        del unet
        self.pipeline.unet = UnetDownSampling.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="unet"
        )
        self.pipeline.scheduler = DDIMScheduler.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="scheduler"
        )

        self.null_inversion = NullTextInversion(self.pipeline)
        self.pipeline.to(accelerator.device)


        optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.learning_rate)
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        self.actor, optimizer, dataloader = accelerator.prepare(
            self.actor, optimizer, dataloader
        )
        global_step = 0
        total_train_steps = self.args.num_epochs * len(dataloader)
        progress_bar = tqdm(range(global_step, total_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        for epoch in range(self.args.num_epochs):
            for step, batch in enumerate(dataloader):
                self.actor.train()
                with accelerator.accumulate(self.actor):
                    image_path = batch["image_path"]
                    src_prompt = batch["src_prompt"]
                    tgt_prompt = batch["tgt_prompt"]
                    uncond_embeddings = batch["null_text_embedding"]
                    batch_size = len(image_path)
                    denoising_trajectories = self.make_random_trajectory(batch_size, accelerator.device)
                    latents = self.ddim_inversion(image_path[0], src_prompt[0])
                    _, hidden_feature_list = self.denoising_trajectory(
                        latents, denoising_trajectories, src_prompt[0], tgt_prompt[0], uncond_embeddings[0]
                    )
                    timesteps = self.pipeline.scheduler.timesteps.to(accelerator.device)
                    timesteps = torch.cat([timesteps] * batch_size).reshape([-1])
                    num_timesteps = hidden_feature_list.shape[1]; channels = hidden_feature_list.shape[2]
                    height = hidden_feature_list.shape[3]; width = hidden_feature_list.shape[4]
                    hidden_feature_list = hidden_feature_list.view(num_timesteps * batch_size, channels, height, width).float()
                    logits_denoising = self.actor(hidden_feature_list, timesteps)
                    denoising_trajectories = denoising_trajectories.view(-1)
                    loss = self.criterion(logits_denoising, denoising_trajectories)
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()
                    if accelerator.is_main_process:
                        global_step += 1
                        logs = {"Epoch": epoch, "loss": loss.detach().item()}
                        progress_bar.update(1)
                        progress_bar.set_postfix(**logs)
                        writer.add_scalar("Loss/loss", loss.item(), global_step)
                    
            if epoch % self.args.checkpointing_steps == 0:
                save_path = os.path.join(self.args.work_dir, f"checkpoint_{epoch}.ckpt")
                self.save_checkpoint(save_path, accelerator)
        if accelerator.is_main_process:
            writer.close()
    
    def save_checkpoint(self, save_path, accelerator):
        unwrap_actor = accelerator.unwrap_model(self.actor)
        torch.save(unwrap_actor.state_dict(), save_path)
    
    def kl_penalized_reward(
        self,
        reward: torch.Tensor,
        log_prob_rl: torch.Tensor,
        log_prob_sft: torch.Tensor,
        action_mask: torch.Tensor = None,
    ) -> Union[torch.Tensor, torch.Tensor]:
        # log(π_RL(y|x) / π_SFL(y|x)) = log(π_RL(y|x)) - log(π_SFL(y|x))
        ratio = log_prob_rl - log_prob_sft
        # k3 in http://joschu.net/blog/kl-approx.html
        estimated_kl = (torch.exp(ratio) - 1) - ratio
        if action_mask:
            estimated_kl = estimated_kl * action_mask
            estimated_kl.sum(dim=1) / action_mask.sum(dim=1)
        estimated_kl = estimated_kl.mean(dim=1, keepdim=True)  # estimated_kl -> (B, 1)
        return reward - self.kl_beta * estimated_kl, estimated_kl
    
    def train_ppo(self):
        logging_dir = Path(self.args.work_dir, "logging")
        accelerator_project_config = ProjectConfiguration(project_dir=self.args.work_dir, logging_dir=logging_dir)

        accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=self.args.mixed_precision,
            log_with=self.args.report_to,
            project_config=accelerator_project_config,
        )
        if accelerator.is_main_process:
            if self.args.work_dir is not None:
                os.makedirs(self.args.work_dir, exist_ok=True)
                os.makedirs(os.path.join(self.args.work_dir, "logging"), exist_ok=True)
            writer = SummaryWriter(log_dir=os.path.join(self.args.work_dir, "logging"))
        
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
        )
        unet = self.pipeline.unet
        del unet
        self.pipeline.unet = UnetDownSampling.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="unet"
        )
        self.pipeline.scheduler = DDIMScheduler.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="scheduler"
        )

        self.null_inversion = NullTextInversion(self.pipeline)
        self.pipeline.to(accelerator.device)

        self.sft_model.requires_grad_(False)
        self.sft_model.to(accelerator.device)

        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                           lr=self.actor_lr,
                                           betas=(self.adam_beta1, self.adam_beta2))
        critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                            lr=self.critic_lr,
                                            betas=(self.adam_beta1, self.adam_beta2))
        self.actor, self.critic, actor_optimizer, critic_optimizer, dataloader = accelerator.prepare(
            self.actor, self.critic, actor_optimizer, critic_optimizer, dataloader
        )
        global_step = 0
        total_train_steps = self.args.num_epochs * len(dataloader)
        progress_bar = tqdm(range(global_step, total_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        for epoch in range(self.args.num_epochs):
            for step, batch in enumerate(dataloader):
                self.actor.train()
                with accelerator.accumulate(self.actor):
                    src_prompts = batch["src_prompt"]
                    tgt_prompts = batch["tgt_prompt"]
                    image_path = batch["image_path"]
                    batch_size = len(src_prompts)
                    latents = self.ddim_inversion(image_path[0], src_prompts[0])
                    uncond_embeddings = batch["null_text_embedding"]
                    masks = batch["masks"]
                    experience = self.make_experience(
                        image_path[0], latents, src_prompts[0], tgt_prompts[0], uncond_embeddings[0], masks, accelerator
                    )

                    self.actor.train()
                    hidden_feature_list = experience.hidden_feature_list
                    batch_size, num_timesteps = hidden_feature_list.shape[:2]
                    timesteps = self.pipeline.scheduler.timesteps
                    timesteps = torch.cat([timesteps]*batch_size)

                    unwrapped_actor = accelerator.unwrap_model(self.actor)

                    actor_log_probs = unwrapped_actor.forward_actor(
                        experience.hidden_feature_list,
                        timesteps,
                        experience.action_list
                    )
                    actor_log_probs = actor_log_probs.squeeze(-1)
                    actor_loss = self.policy_loss(
                        actor_log_probs,
                        experience.log_probs,
                        experience.advantage
                    )
                    params_to_clip = list(self.actor.parameters())
                    accelerator.backward(actor_loss)
                    accelerator.clip_grad_norm_(params_to_clip, 1.0)
                    actor_optimizer.step()
                    actor_optimizer.zero_grad()

                    self.critic.train()
                    unwrapped_critic = accelerator.unwrap_model(self.critic)
                    new_values = unwrapped_critic.forward_critic(
                        experience.hidden_feature_list, timesteps
                    ).view(-1, 1)
                    critic_loss = self.value_loss(
                        new_values,
                        experience.kl_penalized_reward,
                        experience.values
                    )

                    accelerator.backward(critic_loss)
                    params_to_clip = list(self.critic.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, 1.0)
                    critic_optimizer.step()
                    critic_optimizer.zero_grad()
                    if accelerator.is_main_process:
                        global_step += 1
                        logs = {
                            "Epoch": epoch, 
                            "actor_loss": actor_loss.detach().item(),
                            "critic_loss": critic_loss.detach().item(),
                            "rewards": experience.rewards.mean().item(),
                        }
                        progress_bar.update(1)
                        progress_bar.set_postfix(**logs)
                        writer.add_scalar("actor_loss", actor_loss.detach().item(), global_step)
                        writer.add_scalar("critic_loss", critic_loss.detach().item(), global_step)
                        writer.add_scalar("rewards", experience.rewards.mean().item(), global_step)
            if epoch % self.args.checkpointing_steps == 0:
                save_path = os.path.join(self.args.work_dir, f"checkpoint_{epoch}.ckpt")
                self.save_checkpoint_ppo(save_path, accelerator)
            
        if accelerator.is_main_process:
            writer.close()
    
    def save_checkpoint_ppo(self, save_path, accelerator):
        unwrap_actor = accelerator.unwrap_model(self.actor)
        unwrap_critic = accelerator.unwrap_model(self.critic)
        torch.save({
            "actor_state_dict": unwrap_actor.state_dict(),
            "critic_state_dict": unwrap_critic.state_dict(),
        }, save_path)

def generate_with_different_hyperparams(args):
    dataset = EditingDataset(
        annotation_folder=args.annotation_folder,
        annotation_file=args.annotation_file,
        image_size=(512, 512)
    )
    actor = UnetActorDiscrete(in_dim=1280*2, hidden_dim=768, output_dim=2)
    null_text_inversion = NullTextInversionWithStableDiffusion(dataset, actor, args)
    null_text_inversion.init_model()

    os.makedirs(f"saved_results/null_inversion_{args.stop_timestep}", exist_ok=True)

    for data_idx in tqdm(range(len(dataset))):
        image_path = dataset[data_idx]["image_path"]
        src_prompt = dataset[data_idx]["src_prompt"]
        tgt_prompt = dataset[data_idx]["tgt_prompt"]
        null_text_embeddings = dataset[data_idx]["null_text_embedding"]
        latents = null_text_inversion.ddim_inversion(image_path, src_prompt)
        denoising_trajectory = torch.ones((1, NUM_DDIM_STEPS)).long().to("cuda")
        denoising_trajectory[0, :args.stop_timestep] = 0
        latents, _ = null_text_inversion.denoising_trajectory(
            latents, denoising_trajectory, src_prompt, tgt_prompt, null_text_embeddings, verbose=False
        )
        image = null_text_inversion.decode_latents(latents)
        image = Image.fromarray(image[0])
        image.save(f"saved_results/null_inversion_{args.stop_timestep}/edit_{data_idx}.png")

def train_first_stage(args):
    dataset = EditingDataset(
        annotation_folder=args.annotation_folder,
        annotation_file=args.annotation_file,
        image_size=(512, 512)
    )
    actor = UnetActorDiscrete(in_dim=1280*2, hidden_dim=768, output_dim=2)
    null_text_inversion = NullTextInversionWithStableDiffusion(dataset, actor, args)
    null_text_inversion.train_first_stage()

def model_inference(args):
    actor = UnetActorDiscrete(in_dim=1280*2, hidden_dim=768, output_dim=2)
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    actor.load_state_dict(state_dict["actor_state_dict"])
    null_text_inversion = NullTextInversionWithStableDiffusion(None, actor, args, None, None)
    null_text_inversion.init_model()
    null_text = NullTextInversion(null_text_inversion.pipeline)
    image_path = args.input_image_path
    src_prompt = args.src_prompt
    tgt_prompt = args.tgt_prompt

    # null_text_embeddings = torch.load("null_text_embedding.pkl")[1]
    print(f"Invert the null text embeddings ...")
    _,  _, null_text_embeddings = null_text.invert(image_path, src_prompt, offsets=(0,0,0,0))
    null_text_embeddings = torch.stack(null_text_embeddings, dim=0)
    print(f"Generate the edited image ...")
    with torch.no_grad():
        latents = null_text_inversion.ddim_inversion(image_path, src_prompt)
        latents, _, _ = null_text_inversion.batch_generate(
            latents, src_prompt, tgt_prompt, null_text_embeddings, verbose=True
        )
        image = null_text_inversion.decode_latents(latents)
    Image.fromarray(image[0]).save(args.save_edit_path)

def train_ppo(args):
    dataset = EditingDataset(
        annotation_folder=args.annotation_folder,
        annotation_file=args.annotation_file,
        image_size=(512, 512)
    )
    actor = UnetActorDiscrete(in_dim=1280*2, hidden_dim=768, output_dim=2)
    sft_model = UnetActorDiscrete(in_dim=1280*2, hidden_dim=768, output_dim=2)
    actor.load_state_dict(torch.load("exp/sft/checkpoint_7.ckpt", map_location="cpu"))
    sft_model.load_state_dict(torch.load("exp/sft/checkpoint_7.ckpt", map_location="cpu"))
    critic = UnetCriticDiscrete(in_dim=1280*2, hidden_dim=768)
    null_text_inversion = NullTextInversionWithStableDiffusion(dataset, actor, args, sft_model, critic)
    null_text_inversion.train_ppo()

def test_model(args):
    dataset = EditingDataset(
        annotation_folder=args.annotation_folder,
        annotation_file=args.annotation_file,
        image_size=(512, 512)
    )
    actor = UnetActorDiscrete(in_dim=1280*2, hidden_dim=768, output_dim=2)
    null_text_inversion = NullTextInversionWithStableDiffusion(dataset, actor, args)
    null_text_inversion.init_model()

    data_idx = 600

    image_path = dataset[data_idx]["image_path"]
    src_prompt = dataset[data_idx]["src_prompt"]
    tgt_prompt = dataset[data_idx]["tgt_prompt"]
    null_text_embeddings = dataset[data_idx]["null_text_embedding"]
    latents = null_text_inversion.ddim_inversion(image_path, src_prompt)
    denoising_trajectory = torch.ones((1, NUM_DDIM_STEPS)).long().to("cuda")
    denoising_trajectory[0, :50] = 0
    latents, _ = null_text_inversion.denoising_trajectory(
        latents, denoising_trajectory, src_prompt, tgt_prompt, null_text_embeddings, verbose=True
    )
    image = null_text_inversion.decode_latents(latents)
    Image.fromarray(image[0]).save("test.png")
if __name__ == "__main__":
    args = parse_arguments()
    model_inference(args)