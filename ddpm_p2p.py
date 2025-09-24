from prompt_to_prompt.ptp_classes import load_512, AttentionStore
# from ddm_inversion.inversion_utils import inversion_forward_process
from torch import autocast, inference_mode
import torch
from ddm_inversion.inversion_utils import encode_text
import abc
from prompt_to_prompt import seq_aligner
from prompt_to_prompt.ptp_utils import register_attention_control
from tqdm import tqdm
from ddm_inversion.utils import image_grid
from diffusers import StableDiffusionPipeline, DDIMScheduler
from ddpm_inversion_utils import inversion_forward_process, reverse_step
from torch.utils.data import DataLoader
from p2p_dataset import collate_fn
from reward import RewardFunction
import numpy as np
import os
from accelerate.utils import ProjectConfiguration
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss
from accelerate import Accelerator
from torch.nn import functional as F
from typing import Union
from dataclasses import dataclass
from loss import PolicyLoss, ValueLoss
from PIL import Image
from prompt_to_prompt.seq_aligner import get_refinement_mapper
from diffusers import StableDiffusionPipeline, DDIMScheduler
from actor_critic import UnetDownSampling, UnetActorDiscrete, UnetCriticDiscrete

class AlternateAttention(AttentionStore, abc.ABC):

    def __init__(self,
                 cur_cross_trajectories=None,
                 cur_self_trajectories=None,
                 batch_size=None,
                 local_blend=None,
                 mapper=None,
                 alphas=None):
        super().__init__()
        self.cur_cross_trajectories = cur_cross_trajectories
        self.cur_self_trajectories = cur_self_trajectories
        self.batch_size = batch_size
        self.local_blend = local_blend
        self.mapper = mapper
        self.alphas = alphas
    
    def set_trajectories(self, cur_cross_trajectories, cur_self_trajectories, batch_size, mapper, alphas):
        self.cur_cross_trajectories = cur_cross_trajectories
        self.cur_self_trajectories = cur_self_trajectories
        self.batch_size = batch_size
        self.mapper = mapper
        self.alphas = alphas
    
    def refresh_trajectories(self):
        self.cur_cross_trajectories = None
        self.cur_self_trajectories = None
        self.batch_size = None
        self.mapper = None
        self.alphas = None
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
    
    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
    
    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            # return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AlternateAttention, self).forward(attn, is_cross, place_in_unet)
        if (self.cur_cross_trajectories is not None) and (self.cur_self_trajectories is not None):
            if is_cross or (self.cur_self_trajectories==1).all():
                h = attn.shape[0] // (2*self.batch_size)
                attn = attn.reshape(2*self.batch_size, h, *attn.shape[1:])
                attn_base, attn_replace = attn[:self.batch_size], attn[self.batch_size:]
                if is_cross:
                    # attn_base_replace = torch.cat([attn_base[i, :, :, self.mapper[i]] for i in range(len(attn_base))])
                    attn_base_replace = torch.stack([attn_base[i, :, :, self.mapper[i]] for i in range(len(attn_base))])
                    attn_base_replace = attn_base_replace * self.alphas[:, None, None, :] + (1-self.alphas[:, None, None, :]) * attn_replace
                    attn_replace_new = attn_base_replace * self.cur_cross_trajectories + \
                                        (1-self.cur_cross_trajectories) * attn_replace
                    # attn_replace_new = attn_base * self.cur_cross_trajectories + \
                    #                     (1-self.cur_cross_trajectories) * attn_replace
                    attn[self.batch_size:] = attn_replace_new
                else:
                    attn[self.batch_size:] = self.replace_self_attention(attn_base, attn_replace)
                
                attn = attn.reshape(self.batch_size*2*h, *attn.shape[2:])
        
        return attn

class Prompt2PromptDDPM:
    def __init__(self,
                args,
                dataset,
                device,
                ):
        self.args = args
        self.device = device
        self.cfg_scale_src = 3.5
        self.cfg_scale_tgt = 8.0
        self.num_inference_steps=50
        self.eta=1.0
        self.device = device
        self.reward_fn = RewardFunction(alpha1=3.0, alpha2=3.0, device="cuda")
        self.criterion = CrossEntropyLoss()
    
    def init_model(self):
        model_id = "CompVis/stable-diffusion-v1-4"
        self.pipeline = StableDiffusionPipeline.from_pretrained(model_id).to(self.device)
        del self.pipeline.unet
        self.pipeline.unet = UnetDownSampling.from_pretrained(model_id, subfolder="unet").to(self.device)
        self.pipeline.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.pipeline.scheduler.set_timesteps(self.num_inference_steps)
        self.actor = UnetActorDiscrete(1280*2, 768, 2, 2)
        if self.args.checkpoint:
            state_dict = torch.load(self.args.checkpoint, map_location="cpu")
            if "actor" in state_dict:
                self.actor.load_state_dict(state_dict['actor'])
            else:
                self.actor.load_state_dict(state_dict)
        
        self.actor.to(self.device)

        self.controller = AlternateAttention()
        register_attention_control(self.pipeline, self.controller)

    
    @torch.no_grad()
    def denoising_image_batch(self,
                              batch,
                              cross_replace_step,
                              self_replace_step,
                              skip, file_name):
        w0 = (self.pipeline.vae.encode(batch["images"].to(self.device)).latent_dist.mode() * 0.18215).float()
        src_prompt_embedding = self.pipeline.text_encoder(batch["src_input_ids"].to(self.device))[0]
        tgt_prompt_embedding = self.pipeline.text_encoder(batch["tgt_input_ids"].to(self.device))[0]
        batch_size = src_prompt_embedding.shape[0]
        self.controller.reset()
        wt, zs, xts = inversion_forward_process(
            self.pipeline,
            w0,
            src_prompt_embedding,
            etas=self.eta,
            cfg_scale=self.cfg_scale_src,
            num_inference_steps=self.num_inference_steps
        )

        self.controller.reset()
        mapper = batch["mapper"].to(self.device)
        alphas = batch["alphas"].to(self.device)
        denosing_trajectories = torch.zeros((batch_size, self.num_inference_steps)).float().to(self.device)
        denosing_trajectories[:, skip:] = 1
        cross_replace_trajectory = torch.zeros(batch_size, self.num_inference_steps, 77).float().to(self.device)
        cross_replace_step = int(cross_replace_step * self.num_inference_steps)
        cross_replace_trajectory[:, :cross_replace_step, :] = 1.

        self_replace_trajectory = torch.zeros((batch_size, self.num_inference_steps)).float().to(self.device)
        self_replace_step = int(self_replace_step * self.num_inference_steps)
        self_replace_trajectory[:, :self_replace_step] = 1

        src_embeddings = torch.cat([src_prompt_embedding, src_prompt_embedding])
        tgt_embeddings = torch.cat([src_prompt_embedding, tgt_prompt_embedding])

        src_cfg_scales = [self.cfg_scale_src] * batch_size + [self.cfg_scale_src] * batch_size
        src_cfg_scales = torch.Tensor(src_cfg_scales).view(-1, 1, 1,1).to(self.device)
        # tgt_cfg_scales = [self.cfg_scale_src, self.cfg_scale_tgt]
        tgt_cfg_scales = [self.cfg_scale_src] * batch_size + [self.cfg_scale_tgt] * batch_size
        tgt_cfg_scales = torch.Tensor(tgt_cfg_scales).view(-1, 1, 1, 1).to(self.device)
        uncond_embedding = encode_text(self.pipeline, [""] * (batch_size * 2))

        timesteps = self.pipeline.scheduler.timesteps.to(self.device)
        xt = torch.cat([xts[:, -1, :, :]] * 2)
        op = tqdm(timesteps)
        t_to_idx = {int(v):k for k,v in enumerate(timesteps)}

        for i, t in enumerate(op):
            self.controller.set_trajectories(
                cross_replace_trajectory[:,None,i:i+1,:],
                self_replace_trajectory[:, i],
                batch_size,
                mapper,
                alphas
            )
            idx = self.num_inference_steps - t_to_idx[int(t)] - 1
            src_cond_out = self.pipeline.unet.forward(xt, t, encoder_hidden_states=src_embeddings).sample
            tgt_cond_out = self.pipeline.unet.forward(xt, t, encoder_hidden_states=tgt_embeddings).sample
            uncond_out = self.pipeline.unet.forward(xt, t, encoder_hidden_states=uncond_embedding).sample
            # denosing_trajectories = torch.cat([denosing_trajectories] * 2)
            mask = (denosing_trajectories[:,i] == 0).float()
            mask = torch.cat([mask] * 2)
            
            cond_out = mask[:,None,None,None] * src_cond_out + (1-mask[:,None,None,None]) * tgt_cond_out
            cfg_scales = mask[:,None,None,None] * src_cfg_scales + (1-mask[:,None,None,None]) * tgt_cfg_scales
            noise_pred = uncond_out + cfg_scales * (cond_out - uncond_out)
            z = zs[:,idx, :, :, :]
            # z = z.expand(batch_size, -1, -1, -1)
            z = torch.cat([z, z])
            xt = reverse_step(self.pipeline, noise_pred, t, xt, self.eta, variance_noise=z)
            xt = self.controller.step_callback(xt)
        
        xt = xt[batch_size:]
        
        with autocast("cuda"), inference_mode():
            x0_dec = self.pipeline.vae.decode(1 / 0.18215 * xt).sample
        
        img = image_grid(x0_dec)
        img.save(f"{file_name}")
        origin_images = self.convert_image_to_origin_space(batch["images"])
        img = np.array(img)[None,:,:,:]
        edit_mask = batch["masks"].cpu().unsqueeze(-1).numpy()
        reward = self.reward_fn.estimate_reward(origin_images, img, batch["edit_prompts"], edit_mask=edit_mask)
        return reward
    
    @torch.no_grad()
    def convert_image_to_origin_space(self, image):
        image = 127.5*(image + 1).permute(0,2,3,1).cpu().numpy()
        return image.astype(np.uint8)
    
    @torch.no_grad()
    def make_trajectory(self, batch_size, ratio, cross_attention_ratio, self_attention_ratio, num_timesteps):
        denoising_trajectories = torch.zeros((batch_size, num_timesteps)).long().to(self.device)
        skip_timesteps = (ratio * num_timesteps).long()
        for i, skip_timestep in enumerate(skip_timesteps):
            denoising_trajectories[i, skip_timestep:] = 1
        
        cross_replace_trajectories = torch.zeros((batch_size, num_timesteps)).float().to(self.device)
        cross_replace_steps = (cross_attention_ratio * num_timesteps).long()
        for i, cross_replace_step in enumerate(cross_replace_steps):
            cross_replace_trajectories[i, :cross_replace_step] = 1.
        
        self_replace_trajectories = torch.zeros((batch_size, num_timesteps)).float().to(self.device)
        self_replace_steps = (self_attention_ratio * num_timesteps).long()
        for i, self_replace_step in enumerate(self_replace_steps):
            self_replace_trajectories[:, :self_replace_step] = 1.
        
        return denoising_trajectories, cross_replace_trajectories, self_replace_trajectories
    
    @torch.no_grad()
    def denoising_trajectory(self,
                             init_latents,
                             batch,
                             denoising_trajectories,
                             cross_replace_trajectory,
                             self_replace_trajectory,
                             src_prompt_embedding,
                             tgt_prompt_embedding,
                             ):
        batch_size = src_prompt_embedding.shape[0]
        self.controller.reset()
        wt, zs, xts = inversion_forward_process(
            self.pipeline,
            init_latents,
            src_prompt_embedding,
            etas=self.eta,
            cfg_scale=self.cfg_scale_src,
            num_inference_steps=self.num_inference_steps
        )

        self.controller.reset()
        mapper = batch["mapper"].to(self.device)
        alphas = batch["alphas"].to(self.device)
        src_embeddings = torch.cat([src_prompt_embedding, src_prompt_embedding])
        tgt_embeddings = torch.cat([src_prompt_embedding, tgt_prompt_embedding])

        src_cfg_scales = [self.cfg_scale_src] * batch_size + [self.cfg_scale_src] * batch_size
        src_cfg_scales = torch.Tensor(src_cfg_scales).view(-1, 1, 1,1).to(self.device)

        tgt_cfg_scales = [self.cfg_scale_src] * batch_size + [self.cfg_scale_tgt] * batch_size
        tgt_cfg_scales = torch.Tensor(tgt_cfg_scales).view(-1, 1, 1, 1).to(self.device)
        uncond_embedding = encode_text(self.pipeline, [""] * (batch_size * 2))

        timesteps = self.pipeline.scheduler.timesteps.to(self.device)
        xt = torch.cat([xts[:, -1, :, :]] * 2)
        t_to_idx = {int(v):k for k,v in enumerate(timesteps)}

        hidden_feature_list = []

        for i, t in enumerate(tqdm(timesteps)):
            self.controller.set_trajectories(
                None,
                None,
                None,
                mapper,
                alphas
            )
            hidden_features, _ = self.pipeline.unet.forward(xt, t, encoder_hidden_states=tgt_embeddings, return_hidden=True)
            hidden_features_src, hidden_features_tgt = hidden_features[:batch_size], hidden_features[batch_size:]
            hidden_feature_list.append(torch.cat([hidden_features_src, hidden_features_tgt], dim=1))
            
            self.controller.set_trajectories(
                cross_replace_trajectory[:, None, i:i+1, :],
                self_replace_trajectory[:, i:i+1],
                batch_size=batch_size,
                mapper=mapper,
                alphas=alphas
            )
            idx = self.num_inference_steps - t_to_idx[int(t)] - 1
            src_cond_out = self.pipeline.unet.forward(xt, t, encoder_hidden_states=src_embeddings).sample
            tgt_cond_out = self.pipeline.unet.forward(xt, t, encoder_hidden_states=tgt_embeddings).sample
            uncond_out = self.pipeline.unet.forward(xt, t, encoder_hidden_states=uncond_embedding).sample

            mask = (denoising_trajectories[:,i] == 0).float()
            mask = torch.cat([mask] * 2)

            cond_out = mask[:,None,None,None] * src_cond_out + (1-mask[:,None,None,None]) * tgt_cond_out
            cfg_scales = mask[:,None,None,None] * src_cfg_scales + (1-mask[:,None,None,None]) * tgt_cfg_scales
            noise_pred = uncond_out + cfg_scales * (cond_out - uncond_out)

            z = zs[:, idx, :, :, :]
            z = torch.cat([z, z])
            xt = reverse_step(self.pipeline, noise_pred, t, xt, self.eta, variance_noise=z)
            xt = self.controller.step_callback(xt)
        
        xt = xt[batch_size:]
        self.controller.reset()
        hidden_feature_list = torch.stack(hidden_feature_list, dim=1)
        return xt, hidden_feature_list

    
    def train_first_stage(self):
        logging_dir = os.path.join(self.args.exp_name, self.args.logging_dir)
        accelerator_project_config = ProjectConfiguration(project_dir=self.args.exp_name, logging_dir=logging_dir)
        accelerator = Accelerator(
            log_with=self.args.report_to,
            project_config=accelerator_project_config,
        )
        if accelerator.is_main_process:
            if self.args.exp_name is not None:
                os.makedirs(self.args.exp_name, exist_ok=True)
                os.makedirs(os.path.join(self.args.exp_name, self.args.logging_dir), exist_ok=True)
            
            writer = SummaryWriter(os.path.join(self.args.exp_name, self.args.logging_dir))

        # Pipeline initialization
        self.pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        del self.pipeline.unet
        self.pipeline.unet = UnetDownSampling.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
        self.pipeline.scheduler = DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
        self.pipeline.scheduler.set_timesteps(self.num_inference_steps)
        self.actor = UnetActorDiscrete(1280*2, 768, 2, 2)
        self.controller = AlternateAttention()
        register_attention_control(self.pipeline, self.controller)
        
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.unet.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)

        self.pipeline.to(accelerator.device)
        self.pipeline.vae.to(accelerator.device)
        self.pipeline.unet.to(accelerator.device)
        self.pipeline.text_encoder.to(accelerator.device)
        self.actor.to(accelerator.device)

        optimizer = torch.optim.Adam(self.actor.parameters(), 
                                     lr=0.5e-3)
        
        dataloader = DataLoader(
            self.dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn=collate_fn
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
                    src_input_ids = batch["src_input_ids"].to(accelerator.device)
                    tgt_input_ids = batch["tgt_input_ids"].to(accelerator.device)
                    images = batch["images"].to(accelerator.device)

                    src_embeddings = self.pipeline.text_encoder(src_input_ids)[0]
                    tgt_embeddings = self.pipeline.text_encoder(tgt_input_ids)[0]
                    latents = (self.pipeline.vae.encode(images).latent_dist.mode() * 0.18215)
                    batch_size = latents.shape[0]
                    ratio = 0.05+ 0.5*torch.rand((batch_size,))
                    cross_attention_ratio = 0.2 + 0.3*torch.rand((batch_size,))
                    self_attention_ratio = torch.tensor([0.4]*batch_size)

                    denoising_trajectories, cross_replace_trajectories, self_replace_trajectories = \
                        self.make_trajectory(batch_size,
                                             ratio,
                                             cross_attention_ratio,
                                             self_attention_ratio,
                                             self.num_inference_steps)
                    xt, hidden_feature_list = self.denoising_trajectory(latents, batch,
                                                                        denoising_trajectories,
                                                                        cross_replace_trajectories[:,:,None].repeat(1,1,77),
                                                                        self_replace_trajectories,
                                                                        src_embeddings,
                                                                        tgt_embeddings)
                    
                    batch_size, num_timesteps, channels, height, width = hidden_feature_list.shape
                    hidden_feature_list = hidden_feature_list.view(num_timesteps * batch_size, channels, height, width)
                    timesteps = self.pipeline.scheduler.timesteps
                    timesteps = torch.cat([timesteps] * batch_size).reshape([-1])
                    logits_denoising, logit_attention = self.actor(hidden_feature_list, timesteps)
                    denoising_trajectories = denoising_trajectories.view(-1)
                    denoising_loss = self.criterion(logits_denoising, denoising_trajectories)
                    cross_replace_trajectories = cross_replace_trajectories.view(-1).long()
                    attention_loss = self.criterion(logit_attention, cross_replace_trajectories)
                    loss = denoising_loss + attention_loss
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()
                    if accelerator.is_main_process:
                        global_step += 1
                        logs = {"Epoch": epoch, "loss": loss.detach().item(), "denosing_loss": denoising_loss.detach().item(),
                                "attention_loss": attention_loss.detach().item()}
                        progress_bar.update(1)
                        progress_bar.set_postfix(**logs)
                        writer.add_scalar("Loss/loss", loss.item(), global_step)
                    
            if epoch % self.args.checkpointing_steps == 0:
                save_path = os.path.join(self.args.exp_name, f"checkpoint_{epoch}.ckpt")
                self.save_checkpoint(save_path)
        if accelerator.is_main_process:
            writer.close()
        
    def save_checkpoint(self, ckpt_name):
        torch.save(self.actor.state_dict(), ckpt_name)

    
    @torch.no_grad()
    def denosing_image(self, 
                        image_file,
                        src_prompt,
                        tgt_prompt,
                        cross_replace_step,
                        self_replace_step,
                        skip,
                        prog_bar=True):
        offsets=(0,0,0,0)
        x0 = load_512(image_file, *offsets, self.device)
        with autocast("cuda"), inference_mode():
            w0 = (self.pipeline.vae.encode(x0).latent_dist.mode() * 0.18215).float()
        
        src_prompt_embedding = encode_text(self.pipeline, src_prompt)
        tgt_prompt_embedding = encode_text(self.pipeline, tgt_prompt)
        batch_size = src_prompt_embedding.shape[0]
        
        self.controller.reset()
        wt, zs, xts = inversion_forward_process(
            self.pipeline,
            w0,
            src_prompt_embedding,
            etas=self.eta,
            cfg_scale=self.cfg_scale_src,
            num_inference_steps=self.num_inference_steps
        )
        self.controller.reset()
        mapper = torch.arange(0, 77).long().unsqueeze(0).to(self.device)
        alphas = torch.ones((1, 77)).float().to(self.device)
        
        denosing_trajectories = torch.zeros((batch_size, self.num_inference_steps)).float().to(self.device)
        denosing_trajectories[:, skip:] = 1
        cross_replace_trajectory = torch.zeros(batch_size, self.num_inference_steps, 77).float().to(self.device)
        cross_replace_step = int(cross_replace_step * self.num_inference_steps)
        cross_replace_trajectory[:, :cross_replace_step, :] = 1.

        self_replace_trajectory = torch.zeros((batch_size, self.num_inference_steps)).float().to(self.device)
        self_replace_step = int(self_replace_step * self.num_inference_steps)
        self_replace_trajectory[:, :self_replace_step] = 1

        src_embeddings = torch.cat([src_prompt_embedding, src_prompt_embedding])
        tgt_embeddings = torch.cat([src_prompt_embedding, tgt_prompt_embedding])
        # src_cfg_scales = [self.cfg_scale_src, self.cfg_scale_src]
        src_cfg_scales = [self.cfg_scale_src] * batch_size + [self.cfg_scale_src] * batch_size
        src_cfg_scales = torch.Tensor(src_cfg_scales).view(-1, 1, 1,1).to(self.device)
        # tgt_cfg_scales = [self.cfg_scale_src, self.cfg_scale_tgt]
        tgt_cfg_scales = [self.cfg_scale_src] * batch_size + [self.cfg_scale_tgt] * batch_size
        tgt_cfg_scales = torch.Tensor(tgt_cfg_scales).view(-1, 1, 1, 1).to(self.device)
        uncond_embedding = encode_text(self.pipeline, [""] * (batch_size * 2))

        timesteps = self.pipeline.scheduler.timesteps.to(self.device)
        xt = xts[:,-1, :, :].expand(2*batch_size, -1, -1, -1)
        op = tqdm(timesteps) if prog_bar else timesteps
        t_to_idx = {int(v):k for k,v in enumerate(timesteps)}

        for i, t in enumerate(op):
            self.controller.set_trajectories(
                cross_replace_trajectory[:,None,i:i+1,:],
                self_replace_trajectory[:, i],
                batch_size,
                mapper,
                alphas
            )
            idx = self.num_inference_steps - t_to_idx[int(t)] - 1
            src_cond_out = self.pipeline.unet.forward(xt, t, encoder_hidden_states=src_embeddings).sample
            tgt_cond_out = self.pipeline.unet.forward(xt, t, encoder_hidden_states=tgt_embeddings).sample
            uncond_out = self.pipeline.unet.forward(xt, t, encoder_hidden_states=uncond_embedding).sample
            mask = (denosing_trajectories[:,i] == 0).float()
            cond_out = mask[:,None,None,None] * src_cond_out + (1-mask[:,None,None,None]) * tgt_cond_out
            cfg_scales = mask[:,None,None,None] * src_cfg_scales + (1-mask[:,None,None,None]) * tgt_cfg_scales
            noise_pred = uncond_out + cfg_scales * (cond_out - uncond_out)
            z = zs[:,idx, :, :, :]
            z = z.expand(batch_size, -1, -1, -1)
            xt = reverse_step(self.pipeline, noise_pred, t, xt, self.eta, variance_noise=z)
            xt = self.controller.step_callback(xt)
        
        xt = xt[1:]
        
        with autocast("cuda"), inference_mode():
            x0_dec = self.pipeline.vae.decode(1 / 0.18215 * xt).sample
        
        img = image_grid(x0_dec)
        return img
    
    def image_generation(self, 
                         image_path, 
                         src_prompt, 
                         tgt_prompt,
                         prog_bar=True,
                         temperature=1.0):
        self.actor.eval()
        self.actor.to(self.device)
        offsets = (0, 0, 0, 0)
        x0 = load_512(image_path, *offsets, self.device)
        with autocast("cuda"), inference_mode():
            w0 = (self.pipeline.vae.encode(x0).latent_dist.mode() * 0.18215).float()

        src_tar_len_eq = (len(src_prompt.split(" ")) == len(tgt_prompt.split(" ")))
        if src_tar_len_eq:
            mapper = torch.arange(0, 77).long().unsqueeze(0)
            alphas = torch.ones((1, 77)).float()
        else:
            mapper, alphas = get_refinement_mapper([src_prompt, tgt_prompt], self.pipeline.tokenizer)
        
        src_prompt_embedding = encode_text(self.pipeline, src_prompt)
        tgt_prompt_embedding = encode_text(self.pipeline, tgt_prompt)
        batch_size = src_prompt_embedding.shape[0]
        mapper = mapper.to(self.device)
        alphas = alphas.to(self.device)

        ratio = torch.tensor([0.05] * batch_size)
        cross_attention_ratio = torch.tensor([0.2] * batch_size)
        self_attention_ration = torch.tensor([0.4] * batch_size)

        _, _, self_replace_trajectories = self.make_trajectory(
            batch_size, ratio, cross_attention_ratio, self_attention_ration,
            num_timesteps=self.num_inference_steps
        )

        self.controller.reset()
        wt, zs, xts = inversion_forward_process(
            self.pipeline,
            w0,
            src_prompt_embedding,
            etas=self.eta,
            cfg_scale=self.cfg_scale_src,
            num_inference_steps=self.num_inference_steps
        )

        src_embeddings = torch.cat([src_prompt_embedding, src_prompt_embedding])
        tgt_embeddings = torch.cat([src_prompt_embedding, tgt_prompt_embedding])
        src_cfg_scales = [self.cfg_scale_src] * batch_size + [self.cfg_scale_src] * batch_size
        src_cfg_scales = torch.Tensor(src_cfg_scales).view(-1, 1, 1,1).to(self.device)
        tgt_cfg_scales = [self.cfg_scale_src] * batch_size + [self.cfg_scale_tgt] * batch_size
        tgt_cfg_scales = torch.Tensor(tgt_cfg_scales).view(-1, 1, 1, 1).to(self.device)
        uncond_embedding = encode_text(self.pipeline, [""] * (batch_size * 2))

        timesteps = self.pipeline.scheduler.timesteps.to(self.device)
        xt = xts[:,-1, :, :].expand(2*batch_size, -1, -1, -1)
        op = tqdm(timesteps) if prog_bar else timesteps
        t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
        self.controller.reset()
        denoising_list = []
        attention_list = []
        
        for i, t in enumerate(op):
            self.controller.set_trajectories(
                None,
                None,
                None,
                mapper,
                alphas
            )
            hidden_feature_src, _ = self.pipeline.unet(xt, t, encoder_hidden_states=src_embeddings, return_hidden=True)
            hidden_feature_tgt, _ = self.pipeline.unet(xt, t, encoder_hidden_states=tgt_embeddings, return_hidden=True)
            hidden_feature_src = hidden_feature_src[:batch_size]
            hidden_feature_tgt = hidden_feature_tgt[batch_size:]
            hidden_features = torch.cat([hidden_feature_src, hidden_feature_tgt], dim=1)

            timestep = torch.cat([t.unsqueeze(0)] * batch_size)
            logit_denoise, logit_attention = self.actor(hidden_features, timestep)
            logit_denoise = logit_denoise / temperature
            logit_attention = logit_attention / temperature

            prob_denoise = F.softmax(logit_denoise, dim=-1)
            prob_attention = F.softmax(logit_attention, dim=-1)
            next_denoise_action = torch.multinomial(prob_denoise, num_samples=1)
            next_attention_action = torch.multinomial(prob_attention, num_samples=1)

            denoising_list.append(next_denoise_action)
            attention_list.append(next_attention_action)

            # Set up the controller
            cross_replace_trajectory = next_attention_action[:,None,:,None].repeat(1,1,1,77)
            self.controller.set_trajectories(
                cross_replace_trajectory,
                self_replace_trajectories[:,i:i+1],
                batch_size,
                mapper,
                alphas
            )
            idx = self.num_inference_steps - t_to_idx[int(t)] - 1
            src_cond_out = self.pipeline.unet.forward(xt, t, encoder_hidden_states=src_embeddings).sample
            tgt_cond_out = self.pipeline.unet.forward(xt, t, encoder_hidden_states=tgt_embeddings).sample
            uncond_out = self.pipeline.unet.forward(xt, t, encoder_hidden_states=uncond_embedding).sample

            mask = (next_denoise_action[:,:,None,None] == 0).float()
            mask = torch.cat([mask] * 2)
            cond_out = mask * src_cond_out + (1-mask) * tgt_cond_out
            cfg_scales = mask * src_cfg_scales + (1-mask) * tgt_cfg_scales
            noise_pred = uncond_out + cfg_scales * (cond_out - uncond_out)

            z = zs[:, idx, :, :, :]
            z = torch.cat([z, z])
            xt = reverse_step(self.pipeline, noise_pred, t, xt, self.eta, variance_noise=z)
        
        xt = xt[batch_size:]
        denoising_list = torch.cat(denoising_list, dim=1)
        attention_list = torch.cat(attention_list, dim=1)
        with autocast("cuda"), inference_mode():
            x0_dec = self.pipeline.vae.decode(1 / 0.18215 * xt).sample
        
        img = image_grid(x0_dec)
        return img, attention_list

@dataclass
class Experience:
    x0: torch.Tensor
    denoising_list: torch.Tensor
    attention_list: torch.Tensor
    hidden_feature_list: torch.Tensor
    denoising_log_prob: torch.Tensor
    attention_log_prob: torch.Tensor
    denoising_kl_penalized_reward: torch.Tensor
    attention_kl_penalized_reward: torch.Tensor
    denoising_advantage: torch.Tensor
    attention_advantage: torch.Tensor
    denoising_estimate_kl: torch.Tensor
    attention_estimate_kl: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor

class Prompt2PromptDDPMPPO(Prompt2PromptDDPM):
    def __init__(self,
                 args,
                 dataset,
                 device):
        super().__init__(args, dataset, device)
        self.actor_loss = PolicyLoss()
        self.value_loss = ValueLoss()
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.actor_lr = 5e-5
        self.critic_lr = 5e-5
        self.kl_beta = 0.02
    
    @torch.no_grad()
    def batch_generate(
        self,
        images,
        src_prompt_embeddings,
        tgt_prompt_embeddings,
        mapper,
        alphas,
        self_replace_trajectory,
        temperature=1.0
    ):
        batch_size = images.shape[0]
        latents = (self.pipeline.vae.encode(images).latent_dist.mode() * 0.18215)
        uncond_embeddings = encode_text(self.pipeline, [""]*(batch_size * 2))
        self.controller.reset()
        wt, zs, xts = inversion_forward_process(
            self.pipeline,
            latents,
            src_prompt_embeddings,
            etas=self.eta,
            cfg_scale=self.cfg_scale_src,
            num_inference_steps=self.num_inference_steps
        )
        self.controller.reset()
        denoising_list = []
        attention_list = []
        hidden_feature_list = []
        src_embeddings = torch.cat([src_prompt_embeddings, src_prompt_embeddings])
        tgt_embeddings = torch.cat([src_prompt_embeddings, tgt_prompt_embeddings])

        src_cfg_scales = [self.cfg_scale_src] * batch_size + [self.cfg_scale_src] * batch_size
        src_cfg_scales = torch.Tensor(src_cfg_scales).view(-1,1,1,1).to(self.device)

        tgt_cfg_scales = [self.cfg_scale_src] * batch_size + [self.cfg_scale_tgt] * batch_size
        tgt_cfg_scales = torch.Tensor(tgt_cfg_scales).view(-1, 1, 1, 1).to(self.device)

        timesteps = self.pipeline.scheduler.timesteps.to(self.device)
        xt = torch.cat([xts[:, -1, :, :]] * 2)
        t_to_idx = {int(v):k for k,v in enumerate(timesteps)}

        for i, t in enumerate(timesteps):
            self.controller.set_trajectories(
                None,
                None,
                None,
                mapper,
                alphas
            )
            hidden_feature_src, _ = self.pipeline.unet(xt, t, encoder_hidden_states=src_embeddings, return_hidden=True)
            hidden_feature_tgt, _ = self.pipeline.unet(xt, t, encoder_hidden_states=tgt_embeddings, return_hidden=True)
            hidden_feature_src = hidden_feature_src[:batch_size]
            hidden_feature_tgt = hidden_feature_tgt[batch_size:]

            hidden_features = torch.cat([hidden_feature_src, hidden_feature_tgt], dim=1)
            hidden_feature_list.append(hidden_features)
            timestep = torch.cat([t.unsqueeze(0)] * batch_size)
            logit_denoise, logit_attention = self.actor(hidden_features, timestep)
            logit_denoise = logit_denoise / temperature
            logit_attention = logit_attention / temperature

            prob_denoise = F.softmax(logit_denoise, dim=-1)
            prob_attention = F.softmax(logit_attention, dim=-1)
            next_denoise_action = torch.multinomial(prob_denoise, num_samples=1)
            next_attention_action = torch.multinomial(prob_attention, num_samples=1)

            denoising_list.append(next_denoise_action)
            attention_list.append(next_attention_action)

            # Set up the controller
            cross_replace_trajectory = next_attention_action[:,None,:,None].repeat(1,1,1,77)
            self.controller.set_trajectories(
                cross_replace_trajectory,
                self_replace_trajectory[:,i:i+1],
                batch_size,
                mapper,
                alphas
            )
            idx = self.num_inference_steps - t_to_idx[int(t)] - 1
            src_cond_out = self.pipeline.unet.forward(xt, t, encoder_hidden_states=src_embeddings).sample
            tgt_cond_out = self.pipeline.unet.forward(xt, t, encoder_hidden_states=tgt_embeddings).sample
            uncond_out = self.pipeline.unet.forward(xt, t, encoder_hidden_states=uncond_embeddings).sample

            mask = (next_denoise_action[:,:,None,None] == 0).float()
            mask = torch.cat([mask] * 2)
            cond_out = mask * src_cond_out + (1-mask) * tgt_cond_out
            cfg_scales = mask * src_cfg_scales + (1-mask) * tgt_cfg_scales
            noise_pred = uncond_out + cfg_scales * (cond_out - uncond_out)

            z = zs[:, idx, :, :, :]
            z = torch.cat([z, z])
            xt = reverse_step(self.pipeline, noise_pred, t, xt, self.eta, variance_noise=z)
            xt = self.controller.step_callback(xt)
        
        xt = xt[batch_size:]
        hidden_feature_list = torch.stack(hidden_feature_list, dim=1)
        denoising_list = torch.cat(denoising_list, dim=-1).long()
        attention_list = torch.cat(attention_list, dim=-1).long()
        return xt, hidden_feature_list, denoising_list, attention_list
    
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
    
    
    @torch.no_grad()
    def make_experience(self,
                        images,
                        src_embeddings,
                        tgt_embeddings,
                        mapper,
                        alphas,
                        self_replace_trajectory,
                        editing_region_masks,
                        edit_prompts):
        self.sft_model.eval()
        self.actor.eval()
        self.critic.eval()

        x0, hidden_feature_list, denoising_list, attention_list = self.batch_generate(
            images,
            src_embeddings,
            tgt_embeddings,
            mapper,
            alphas,
            self_replace_trajectory
        )
        batch_size = hidden_feature_list.shape[0]

        edit_imgs = self.pipeline.vae.decode(x0/0.18215).sample
        edit_imgs = torch.clamp(edit_imgs, -1., 1.)
        edit_imgs = 127.5*(edit_imgs + 1).permute(0,2,3,1).cpu().numpy()
        edit_imgs = edit_imgs.astype(np.uint8)
        timesteps = self.pipeline.scheduler.timesteps
        timesteps = torch.cat([timesteps] * batch_size)

        denoising_log_prob, attention_log_prob = self.actor.forward_actor(
            hidden_feature_list, timesteps, denoising_list, attention_list
        )
        sft_denoising_log_prob, sft_attention_log_prob = self.sft_model.forward_actor(
            hidden_feature_list, timesteps, denoising_list, attention_list
        )
        values = self.critic.forward_critic(hidden_feature_list, timesteps)

        origin_images = self.convert_image_to_origin_space(images)
        editing_region_masks = editing_region_masks.unsqueeze(-1).cpu().numpy()
        reward = self.reward_fn.estimate_reward(
            origin_images,
            edit_imgs,
            edit_prompts,
            editing_region_masks
        )

        reward = reward.to(self.device).unsqueeze(-1)
        denoising_log_prob = denoising_log_prob.squeeze(-1); attention_log_prob = attention_log_prob.squeeze(-1)
        sft_denoising_log_prob = sft_denoising_log_prob.squeeze(-1); sft_attention_log_prob = sft_attention_log_prob.squeeze(-1)
        denoising_kl_penalized_reward, denoising_estimate_kl = \
            self.kl_penalized_reward(
                reward,
                denoising_log_prob,
                sft_denoising_log_prob
            )
        attention_kl_penalized_reward, attention_estimate_kl = \
            self.kl_penalized_reward(
                reward,
                attention_log_prob,
                sft_attention_log_prob
            )
        
        denoising_advantage = denoising_kl_penalized_reward - values
        attention_advantage = attention_kl_penalized_reward - values
        return Experience(
            x0,
            denoising_list,
            attention_list,
            hidden_feature_list,
            denoising_log_prob,
            attention_log_prob,
            denoising_kl_penalized_reward,
            attention_kl_penalized_reward,
            denoising_advantage,
            attention_advantage,
            denoising_estimate_kl,
            attention_estimate_kl,
            values,
            reward
        )
    
    def fit_ppo(self):
        logging_dir = os.path.join(self.args.exp_name, self.args.logging_dir)
        accelerator_project_config = ProjectConfiguration(project_dir=self.args.exp_name, logging_dir=logging_dir)
        accelerator = Accelerator(
            log_with=self.args.report_to,
            project_config=accelerator_project_config,
        )

        if accelerator.is_main_process:
            if self.args.exp_name is not None:
                os.makedirs(self.args.exp_name, exist_ok=True)
                os.makedirs(os.path.join(self.args.exp_name, self.args.logging_dir), exist_ok=True)
            writer = SummaryWriter(os.path.join(self.args.exp_name, self.args.logging_dir))
        
        self.pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        del self.pipeline.unet
        self.pipeline.unet = UnetDownSampling.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
        self.pipeline.scheduler = DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
        self.pipeline.scheduler.set_timesteps(self.num_inference_steps)
        self.actor = UnetActorDiscrete(1280*2, 768, 2, 2)
        self.sft_model = UnetActorDiscrete(1280*2, 768, 2, 2)
        self.critic = UnetCriticDiscrete(1280*2, 768)
        if self.args.checkpoint:
            self.actor.load_state_dict(torch.load(self.args.checkpoint, map_location="cpu"))
            self.sft_model.load_state_dict(torch.load(self.args.checkpoint, map_location="cpu"))
        
        self.controller = AlternateAttention()
        register_attention_control(self.pipeline, self.controller)
        

        self.pipeline.vae.requires_grad_(False)
        self.pipeline.unet.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)
        self.sft_model.requires_grad_(False)
        
        self.pipeline.to(accelerator.device)
        self.pipeline.vae.to(accelerator.device)
        self.pipeline.unet.to(accelerator.device)
        self.pipeline.text_encoder.to(accelerator.device)

        self.actor.to(accelerator.device)
        self.critic.to(accelerator.device)
        self.sft_model.to(accelerator.device)

        actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                           lr=self.actor_lr,
                                           betas=(self.adam_beta1, self.adam_beta2))
        critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                            lr=self.critic_lr,
                                            betas=(self.adam_beta1, self.adam_beta2))
        self.actor, self.critic, actor_optimizer, critic_optimizer, self.dataloader = accelerator.prepare(
            self.actor, self.critic, actor_optimizer, critic_optimizer, self.dataloader
        )
        global_step = 0
        total_train_steps = self.args.num_epochs * len(self.dataloader)
        progress_bar = tqdm(range(global_step, total_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")
        for epoch in range(self.args.num_epochs):
            for step, batch in enumerate(self.dataloader):
                with accelerator.accumulate(self.actor):
                    src_input_ids = batch["src_input_ids"].to(accelerator.device)
                    tgt_input_ids = batch["tgt_input_ids"].to(accelerator.device)
                    src_embeddings = self.pipeline.text_encoder(src_input_ids)[0]
                    tgt_embeddings = self.pipeline.text_encoder(tgt_input_ids)[0]
                    images = batch["images"].to(accelerator.device)
                    editing_region_masks = batch["masks"].to(accelerator.device)
                    edit_prompts = batch["edit_prompts"]
                    mapper = batch["mapper"].to(accelerator.device)
                    alphas = batch["alphas"].to(accelerator.device)
                    batch_size = images.shape[0]
                    ratio = torch.tensor([0.05] * batch_size)
                    cross_attention_ratio = torch.tensor([0.2] * batch_size)
                    self_attention_ration = torch.tensor([0.4] * batch_size)

                    _, _, self_replace_trajectories = self.make_trajectory(
                        batch_size, ratio, cross_attention_ratio, self_attention_ration,
                        num_timesteps=self.num_inference_steps
                    )

                    experience = self.make_experience(
                        images,
                        src_embeddings,
                        tgt_embeddings,
                        mapper,
                        alphas,
                        self_replace_trajectories,
                        editing_region_masks,
                        edit_prompts
                    )
                    self.actor.train()

                    hidden_feature_list = experience.hidden_feature_list
                    batch_size, num_timesteps = hidden_feature_list.shape[:2]
                    timesteps = self.pipeline.scheduler.timesteps
                    timesteps = torch.cat([timesteps]*batch_size)

                    denoising_actor_log_probs, attention_actor_log_probs = self.actor.forward_actor(
                        experience.hidden_feature_list,
                        timesteps,
                        experience.denoising_list,
                        experience.attention_list
                    )
                    denoising_actor_log_probs = denoising_actor_log_probs.squeeze(-1)
                    denoising_actor_loss = self.actor_loss(
                        denoising_actor_log_probs,
                        experience.denoising_log_prob,
                        experience.denoising_advantage
                    )
                    attention_actor_log_probs = attention_actor_log_probs.squeeze(-1)
                    attention_actor_loss = self.actor_loss(
                        attention_actor_log_probs,
                        experience.attention_log_prob,
                        experience.attention_advantage
                    )
                    actor_loss = denoising_actor_loss + attention_actor_loss
                    params_to_clip = list(self.actor.parameters())
                    accelerator.backward(actor_loss)
                    accelerator.clip_grad_norm_(params_to_clip, 1.0)
                    actor_optimizer.step()
                    actor_optimizer.zero_grad()

                    self.critic.train()
                    new_values = self.critic.forward_critic(
                        experience.hidden_feature_list, timesteps
                    ).view(-1, 1)
                    critic_loss = self.value_loss(
                        new_values,
                        experience.denoising_kl_penalized_reward,
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
                            "denoising_actor_loss": denoising_actor_loss.detach().item(),
                            "attention_actor_loss": attention_actor_loss.detach().item(),
                            "critic_loss": critic_loss.detach().item(),
                            "reward": experience.rewards.mean().item(),
                        }
                        progress_bar.update(1)
                        progress_bar.set_postfix(**logs)
                        writer.add_scalar("actor_loss", actor_loss.detach().item(), global_step)
                        writer.add_scalar("critic_loss", critic_loss.detach().item(), global_step)
                        writer.add_scalar("reward", experience.rewards.mean().item(), global_step)

                
            if epoch % self.args.checkpointing_steps == 0:
                save_path = os.path.join(self.args.exp_name, f"checkpoint_{epoch}.ckpt")
                self.save_checkpoint(save_path)
            
        if accelerator.is_main_process:
            writer.close()
    
    def save_checkpoint(self, checkpoint_path):
        actor_state_dict = self.actor.state_dict()
        critic_state_dict = self.critic.state_dict()
        model = {
            "actor": actor_state_dict,
            "critic": critic_state_dict
        }
        torch.save(model, checkpoint_path)


if __name__ == "__main__":
    pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cuda")
    pipeline.scheduler = DDIMScheduler.from_config("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    p2p_ddpm = Prompt2PromptDDPM(None, pipeline, "cuda")
    image_file = "./example_images/horse_mud.jpg"
    src_prompt = "a photo of a horse in the mud"
    tgt_prompt = "a photo of a horse in the snow"
    p2p_ddpm.denosing_image(image_file, src_prompt, tgt_prompt, cross_replace_step=0.8, self_replace_step=0.3, skip=12)