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
from autoedit_dataset import collate_fn
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
from actor_critic import UnetDownSampling, UnetActorDiscrete, UnetCriticDiscrete

@dataclass
class Experience:
    x0: torch.Tensor
    denoising_list: torch.Tensor
    hidden_feature_list: torch.Tensor
    denoising_log_prob: torch.Tensor
    denoising_kl_penalized_reward: torch.Tensor
    denoising_advantage: torch.Tensor
    denoising_estimate_kl: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor

class Prompt2PromptDDPM:
    def __init__(self,
                args,
                dataset,
                device,
                alpha1,
                alpha2,
                ):
        self.args = args
        self.device = device
        self.cfg_scale_src = 3.5
        self.cfg_scale_tgt = 8.0
        self.num_inference_steps=50
        self.reward_fn = RewardFunction(alpha1=alpha1, alpha2=alpha2, device="cpu")
        self.eta=1.0
        self.dataset = dataset
        self.criterion = CrossEntropyLoss()
    
    def init_model(self):
        model_id = "CompVis/stable-diffusion-v1-4"
        self.pipeline = StableDiffusionPipeline.from_pretrained(model_id).to(self.device)
        self.pipeline.unet = UnetDownSampling.from_pretrained(model_id, subfolder="unet").to(self.device)
        self.pipeline.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.pipeline.scheduler.set_timesteps(self.num_inference_steps)
        self.actor = UnetActorDiscrete(1280*2, 768, 2, 2)
        if self.args.checkpoint:
            state_dict = torch.load(self.args.checkpoint, map_location=self.device)
            if "actor" in state_dict:
                self.actor.load_state_dict(state_dict["actor"])
            else:
                self.actor.load_state_dict(state_dict)
    
    @torch.no_grad()
    def convert_image_to_origin_space(self, image):
        image = 127.5*(image + 1).permute(0,2,3,1).cpu().numpy()
        return image.astype(np.uint8)
    
    @torch.no_grad()
    def make_trajectory(self, batch_size, ratio, num_timesteps):
        denoising_trajectories = torch.zeros((batch_size, num_timesteps)).long().to(self.device)
        skip_timesteps = (ratio * num_timesteps).long()
        for i, skip_timestep in enumerate(skip_timesteps):
            denoising_trajectories[i, skip_timestep:] = 1
        
        return denoising_trajectories
    
    @torch.no_grad()
    def denoising_trajectory(self,
                             init_latents,
                             denoising_trajectories,
                             src_prompt_embedding,
                             tgt_prompt_embedding,
                             ):
        batch_size = src_prompt_embedding.shape[0]
        wt, zs, xts = inversion_forward_process(
            self.pipeline,
            init_latents,
            src_prompt_embedding,
            etas=self.eta,
            cfg_scale=self.cfg_scale_src,
            num_inference_steps=self.num_inference_steps
        )

        uncond_embedding = encode_text(self.pipeline, [""] * batch_size)

        timesteps = self.pipeline.scheduler.timesteps.to(init_latents.device)
        xt = xts[:, -1, :, :, :]
        t_to_idx = {int(v):k for k,v in enumerate(timesteps)}

        hidden_feature_list = []

        for i, t in enumerate(timesteps):
            hidden_features_tgt, tgt_cond_out = self.pipeline.unet.forward(
                xt, 
                t, 
                encoder_hidden_states=tgt_prompt_embedding, 
                return_hidden=True)
            hidden_features_src, src_cond_out = self.pipeline.unet.forward(xt, 
                                                                t,
                                                                encoder_hidden_states=src_prompt_embedding,
                                                                return_hidden=True)
            tgt_cond_out = tgt_cond_out.sample
            src_cond_out = src_cond_out.sample

            hidden_feature_list.append(torch.cat([hidden_features_src, hidden_features_tgt], dim=1))
            idx = self.num_inference_steps - t_to_idx[int(t)] - 1
            
            uncond_out = self.pipeline.unet.forward(xt, t, encoder_hidden_states=uncond_embedding).sample

            mask = (denoising_trajectories[:,i] == 0).float()
            src_noise_pred = uncond_out + self.cfg_scale_src * (src_cond_out - uncond_out)
            tgt_noise_pred = uncond_out + self.cfg_scale_tgt * (tgt_cond_out - uncond_out)
            
            noise_pred = src_noise_pred * mask[:,None,None,None] + (1-mask[:,None,None,None]) * tgt_noise_pred

            z = zs[:, idx, :, :, :]
            xt = reverse_step(self.pipeline, noise_pred, t, xt, self.eta, variance_noise=z)
    
        hidden_feature_list = torch.stack(hidden_feature_list, dim=1)
        return xt, hidden_feature_list
    
    def denoising_image(self, image_path, src_prompt, tgt_prompt, skip):
        offsets=(0,0,0,0)
        x0 = load_512(image_path, *offsets, self.device)
        with autocast("cuda"), inference_mode():
            w0 = (self.pipeline.vae.encode(x0).latent_dist.mode() * 0.18215).float()
        src_prompt_embedding = encode_text(self.pipeline, src_prompt)
        tgt_prompt_embedding = encode_text(self.pipeline, tgt_prompt)
        batch_size = src_prompt_embedding.shape[0]

        wt, zs, xts = inversion_forward_process(
            self.pipeline,
            w0,
            src_prompt_embedding,
            etas=self.eta,
            cfg_scale=self.cfg_scale_src,
            num_inference_steps=self.num_inference_steps
        )

        denoising_trajectories = torch.zeros((batch_size, self.num_inference_steps)).long().to(self.device)
        denoising_trajectories[:, skip:] = 1
        x0, _ = self.denoising_trajectory(
            w0, denoising_trajectories, src_prompt_embedding, tgt_prompt_embedding
        )
        with autocast("cuda"), inference_mode():
            x0_dec = self.pipeline.vae.decode(1 / 0.18215 * x0).sample
        
        img = image_grid(x0_dec)
        return img
    
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
        
        self.pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        self.actor = UnetActorDiscrete(1280*2, 768, 2, 2)
        self.pipeline.unet = UnetDownSampling.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
        self.pipeline.scheduler = DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
        self.pipeline.scheduler.set_timesteps(self.num_inference_steps)
        
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.unet.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)

        self.pipeline.to(accelerator.device)
        self.actor.to(accelerator.device)

        optimizer = torch.optim.Adam(self.actor.parameters(), 
                                     lr=1e-4)
        
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
                    ratio = 0.2 + 0.3*torch.rand((batch_size,))
                    # cross_attention_ratio = 0.2 + 0.3*torch.rand((batch_size,))
                    # self_attention_ratio = torch.tensor([0.4]*batch_size)

                    denoising_trajectories = \
                        self.make_trajectory(batch_size,
                                             ratio,
                                             self.num_inference_steps)
                    xt, hidden_feature_list = self.denoising_trajectory(latents,
                                                                        denoising_trajectories,
                                                                        # cross_replace_trajectories[:,:,None].repeat(1,1,77),
                                                                        # self_replace_trajectories,
                                                                        src_embeddings,
                                                                        tgt_embeddings)
                    
                    batch_size, num_timesteps, channels, height, width = hidden_feature_list.shape
                    hidden_feature_list = hidden_feature_list.view(num_timesteps * batch_size, channels, height, width)
                    timesteps = self.pipeline.scheduler.timesteps
                    timesteps = torch.cat([timesteps] * batch_size).reshape([-1])
                    logits_denoising, _ = self.actor(hidden_feature_list, timesteps)
                    denoising_trajectories = denoising_trajectories.view(-1)
                    denoising_loss = self.criterion(logits_denoising, denoising_trajectories)
                    # cross_replace_trajectories = cross_replace_trajectories.view(-1).long()
                    # attention_loss = self.criterion(logit_attention, cross_replace_trajectories)
                    # loss = denoising_loss + attention_loss
                    loss = denoising_loss
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
                save_path = os.path.join(self.args.exp_name, f"checkpoint_{epoch}.ckpt")
                self.save_checkpoint(save_path, accelerator)
        if accelerator.is_main_process:
            writer.close()
        
    def save_checkpoint(self, ckpt_name, accelerator):
        if accelerator.is_main_process:
            if accelerator is not None:
                unwrapped_model = accelerator.unwrap_model(self.actor)
            else:
                unwrapped_model = self.actor
        torch.save(unwrapped_model.state_dict(), ckpt_name)
    
    @torch.no_grad()
    def image_generation(self,
                         image_path,
                         src_prompt,
                         tgt_prompt,
                         prog_bar=True,
                         temperature=1.0):
        self.actor.eval()
        self.actor.to(self.device)
        offsets = (0,0,0,0)
        x0 = load_512(image_path, *offsets, self.device)
        latents = (self.pipeline.vae.encode(x0).latent_dist.mode() * 0.18215).float()
        src_prompt_embedding = encode_text(self.pipeline, src_prompt)
        tgt_prompt_embedding = encode_text(self.pipeline, tgt_prompt)
        uncond_embeddings = encode_text(self.pipeline, [""] * src_prompt_embedding.shape[0])
        batch_size = src_prompt_embedding.shape[0]
        
        wt, zs, xts = inversion_forward_process(
            self.pipeline,
            latents,
            src_prompt_embedding,
            etas=self.eta,
            cfg_scale=self.cfg_scale_src,
            num_inference_steps=self.num_inference_steps
        )

        timesteps = self.pipeline.scheduler.timesteps.to(self.device)
        xt = xts[:, -1, :, :, :]
        t_to_idx = {int(v):k for k,v in enumerate(timesteps)}
        action_list = []
        op = tqdm(timesteps) if prog_bar else timesteps
        for i, t in enumerate(op):
            hidden_features_tgt, tgt_cond_out = self.pipeline.unet.forward(
                xt,
                t,
                encoder_hidden_states=tgt_prompt_embedding,
                return_hidden=True
            )
            hidden_features_src, src_cond_out = self.pipeline.unet.forward(
                xt,
                t,
                encoder_hidden_states=src_prompt_embedding,
                return_hidden=True
            )

            tgt_cond_out = tgt_cond_out.sample
            src_cond_out = src_cond_out.sample
            uncond_out = self.pipeline.unet(xt,t, encoder_hidden_states=uncond_embeddings).sample
            
            hidden_features = torch.cat([hidden_features_src, hidden_features_tgt], dim=1)
            idx = self.num_inference_steps - t_to_idx[int(t)] - 1
            timesteps = torch.cat([t.unsqueeze(0)] * batch_size)
            logit_denoise, _ = self.actor(hidden_features, timesteps)
            logit_denoise = logit_denoise / temperature
            prob_denoise = F.softmax(logit_denoise, dim=-1)
            next_denoise_action = torch.multinomial(prob_denoise, num_samples=1)

            action_list.append(next_denoise_action)
            mask = (next_denoise_action[:,:,None,None] == 0).float()
            src_noise_pred = uncond_out + self.cfg_scale_src * (src_cond_out - uncond_out)
            tgt_noise_pred = uncond_out + self.cfg_scale_tgt * (tgt_cond_out - uncond_out)
            noise_pred = src_noise_pred * mask + (1-mask) * tgt_noise_pred
            z = zs[:, idx, :, :, :]
            xt = reverse_step(self.pipeline, noise_pred, t, xt, self.eta, variance_noise=z)
        
        action_list = torch.cat(action_list, dim=-1).long()
        
        with autocast("cuda"), inference_mode():
            x0_dec = self.pipeline.vae.decode(1 / 0.18215 * xt).sample
        
        img = image_grid(x0_dec)
        return img, action_list

class Prompt2PromptDDPMPPO(Prompt2PromptDDPM):
    def __init__(self,
                 args,
                 dataset,
                 device,
                 alpha1,
                 alpha2):
        super().__init__(args, dataset, device, alpha1, alpha2)
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
        src_prompt_embedding,
        tgt_prompt_embedding,
        temperature=1.0
    ):
        batch_size = images.shape[0]
        latents = (self.pipeline.vae.encode(images).latent_dist.mode() * 0.18215)
        uncond_embeddings = encode_text(self.pipeline, [""] * batch_size)
        wt, zs, xts = inversion_forward_process(
            self.pipeline,
            latents,
            src_prompt_embedding,
            etas=self.eta,
            cfg_scale=self.cfg_scale_src,
            num_inference_steps=self.num_inference_steps
        )

        timesteps = self.pipeline.scheduler.timesteps.to(self.device)
        xt = xts[:, -1, :, :, :]
        t_to_idx = {int(v):k for k,v in enumerate(timesteps)}

        hidden_feature_list = []
        action_list = []
        for i, t in enumerate(timesteps):
            hidden_features_tgt, tgt_cond_out = self.pipeline.unet.forward(
                xt, 
                t, 
                encoder_hidden_states=tgt_prompt_embedding, 
                return_hidden=True)
            hidden_features_src, src_cond_out = self.pipeline.unet.forward(xt, 
                                                                t,
                                                                encoder_hidden_states=src_prompt_embedding,
                                                                return_hidden=True)
            tgt_cond_out = tgt_cond_out.sample
            src_cond_out = src_cond_out.sample

            hidden_feature_list.append(torch.cat([hidden_features_src, hidden_features_tgt], dim=1))
            idx = self.num_inference_steps - t_to_idx[int(t)] - 1

            uncond_out = self.pipeline.unet(xt,t, encoder_hidden_states=uncond_embeddings).sample

            hidden_features = torch.cat([hidden_features_src, hidden_features_tgt], dim=1)
            timesteps = torch.cat([t.unsqueeze(0)] * batch_size)
            logit_denoise, _ = self.actor(hidden_features, timesteps)
            logit_denoise = logit_denoise / temperature
            prob_denoise = F.softmax(logit_denoise, dim=-1)
            next_denoise_action = torch.multinomial(prob_denoise, num_samples=1)

            action_list.append(next_denoise_action)
            mask = (next_denoise_action[:,:,None,None] == 0).float()
            src_noise_pred = uncond_out + self.cfg_scale_src * (src_cond_out - uncond_out)
            tgt_noise_pred = uncond_out + self.cfg_scale_tgt * (tgt_cond_out - uncond_out)
            noise_pred = src_noise_pred * mask + (1-mask) * tgt_noise_pred

            z = zs[:, idx, :, :, :]
            xt = reverse_step(self.pipeline, noise_pred, t, xt, self.eta, variance_noise=z)
        
        hidden_feature_list = torch.stack(hidden_feature_list, dim=1)
        action_list = torch.cat(action_list, dim=-1).long()
        return xt, hidden_feature_list, action_list
    
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
                        editing_region_masks,
                        edit_prompts,
                        accelerator=None):
        self.sft_model.eval()
        self.actor.eval()
        self.critic.eval()
        x0, hidden_feature_list, denoising_list = self.batch_generate(
            images,
            src_embeddings,
            tgt_embeddings
        )
        batch_size = hidden_feature_list.shape[0]

        edit_imgs = self.pipeline.vae.decode(x0/0.18215).sample
        edit_imgs = torch.clamp(edit_imgs, -1., 1.)
        edit_imgs = 127.5*(edit_imgs + 1).permute(0,2,3,1).cpu().numpy()
        edit_imgs = edit_imgs.astype(np.uint8)
        timesteps = self.pipeline.scheduler.timesteps
        timesteps = torch.cat([timesteps] * batch_size)

        if accelerator is not None:
            denoising_log_prob, _ = accelerator.unwrap_model(self.actor).forward_actor(
                hidden_feature_list, timesteps, denoising_list, None
            )
            sft_denoising_log_prob, _ = accelerator.unwrap_model(self.sft_model).forward_actor(
                hidden_feature_list, timesteps, denoising_list, None
            )
            values = accelerator.unwrap_model(self.critic).forward_critic(hidden_feature_list, timesteps)
        else:
            denoising_log_prob, _ = self.actor.forward_actor(
                hidden_feature_list, timesteps, denoising_list, None
            )
            sft_denoising_log_prob, _ = self.sft_model.forward_actor(
                hidden_feature_list, timesteps, denoising_list, None
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

        reward = reward.to(src_embeddings.device).unsqueeze(-1)
        denoising_log_prob = denoising_log_prob.squeeze(-1)
        sft_denoising_log_prob = sft_denoising_log_prob.squeeze(-1)
        denoising_kl_penalized_reward, denoising_estimate_kl = \
            self.kl_penalized_reward(
                reward,
                denoising_log_prob,
                sft_denoising_log_prob
            )
        
        denoising_advantage = denoising_kl_penalized_reward - values
        return Experience(
            x0,
            denoising_list,
            hidden_feature_list,
            denoising_log_prob,
            denoising_kl_penalized_reward,
            denoising_advantage,
            denoising_estimate_kl,
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
        self.pipeline.unet = UnetDownSampling.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
        self.pipeline.scheduler = DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
        self.pipeline.scheduler.set_timesteps(self.num_inference_steps)

        self.actor = UnetActorDiscrete(1280*2, 768, 2, 2)
        self.sft_model = UnetActorDiscrete(1280*2, 768, 2, 2)
        self.critic = UnetCriticDiscrete(1280*2, 768)
        if self.args.checkpoint:
            self.actor.load_state_dict(torch.load(self.args.checkpoint)["actor"])
            self.sft_model.load_state_dict(torch.load(self.args.checkpoint)["actor"])

        self.pipeline.vae.requires_grad_(False)
        self.pipeline.unet.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)
        self.sft_model.requires_grad_(False)

        self.pipeline.to(accelerator.device)

        self.actor.to(accelerator.device)
        self.critic.to(accelerator.device)
        self.sft_model.to(accelerator.device)

        actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                           lr=self.actor_lr,
                                           betas=(self.adam_beta1, self.adam_beta2))
        critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                            lr=self.critic_lr,
                                            betas=(self.adam_beta1, self.adam_beta2))
        dataloader = DataLoader(
            self.dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn=collate_fn
        )

        self.actor, self.critic, actor_optimizer, critic_optimizer, dataloader = accelerator.prepare(
            self.actor, self.critic, actor_optimizer, critic_optimizer, dataloader
        )
        global_step = 0
        total_train_steps = self.args.num_epochs * len(dataloader)
        progress_bar = tqdm(range(global_step, total_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")
        for epoch in range(self.args.num_epochs):
            for step, batch in enumerate(dataloader):
                with accelerator.accumulate(self.actor):
                    src_input_ids = batch["src_input_ids"].to(accelerator.device)
                    tgt_input_ids = batch["tgt_input_ids"].to(accelerator.device)
                    src_embeddings = self.pipeline.text_encoder(src_input_ids)[0]
                    tgt_embeddings = self.pipeline.text_encoder(tgt_input_ids)[0]
                    images = batch["images"].to(accelerator.device)
                    editing_region_masks = batch["masks"].to(accelerator.device)
                    edit_prompts = batch["edit_prompts"]

                    experience = self.make_experience(
                        images,
                        src_embeddings,
                        tgt_embeddings,
                        editing_region_masks,
                        edit_prompts,
                        accelerator
                    )
                    self.actor.train()

                    hidden_feature_list = experience.hidden_feature_list
                    batch_size, num_timesteps = hidden_feature_list.shape[:2]
                    timesteps = self.pipeline.scheduler.timesteps
                    timesteps = torch.cat([timesteps]*batch_size)

                    denoising_actor_log_probs, _ = accelerator.unwrap_model(self.actor).forward_actor(
                        experience.hidden_feature_list,
                        timesteps,
                        experience.denoising_list,
                        None,
                    )
                    denoising_actor_log_probs = denoising_actor_log_probs.squeeze(-1)
                    actor_loss = self.actor_loss(
                        denoising_actor_log_probs,
                        experience.denoising_log_prob,
                        experience.denoising_advantage
                    )
                    
                    params_to_clip = list(self.actor.parameters())
                    accelerator.backward(actor_loss)
                    accelerator.clip_grad_norm_(params_to_clip, 1.0)
                    actor_optimizer.step()
                    actor_optimizer.zero_grad()

                    self.critic.train()
                    new_values = accelerator.unwrap_model(self.critic).forward_critic(
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
                            "critic_loss": critic_loss.detach().item()
                        }
                        progress_bar.update(1)
                        progress_bar.set_postfix(**logs)
                        writer.add_scalar("actor_loss", actor_loss.detach().item(), global_step)
                        writer.add_scalar("critic_loss", critic_loss.detach().item(), global_step)
                        # if global_step % 50 == 0:
                        #     save_path = os.path.join(self.args.exp_name, f"checkpoint_{global_step}.ckpt")
                        #     self.save_checkpoint(save_path)
            if epoch % self.args.checkpointing_steps == 0:
                save_path = os.path.join(self.args.exp_name, f"checkpoint_{epoch}.ckpt")
                self.save_checkpoint(save_path, accelerator)
            
        if accelerator.is_main_process:
            writer.close()
    
    def save_checkpoint(self, checkpoint_path, accelerator):
        if accelerator is not None:
            if accelerator.is_main_process:
                unwrapped_actor = accelerator.unwrap_model(self.actor)
                unwrapped_critic = accelerator.unwrap_model(self.critic)
            else:
                unwrapped_actor = self.actor
                unwrapped_critic = self.critic
        actor_state_dict = unwrapped_actor.state_dict()
        critic_state_dict = unwrapped_critic.state_dict()
        model = {
            "actor": actor_state_dict,
            "critic": critic_state_dict
        }
        torch.save(model, checkpoint_path)

class Prompt2PromptDDPMPPOWithoutSFT(Prompt2PromptDDPMPPO):
    @torch.no_grad()
    def make_experience(
            self,
            images,
            src_embeddings,
            tgt_embeddings,
            editing_region_masks,
            edit_prompts,
            accelerator=None
    ):
        self.actor.eval()
        self.critic.eval()
        x0, hidden_feature_list, denoising_list = self.batch_generate(
            images,
            src_embeddings,
            tgt_embeddings
        )
        batch_size = hidden_feature_list.shape[0]
        edit_imgs = self.pipeline.vae.decode(x0/0.18215).sample
        edit_imgs = torch.clamp(edit_imgs, -1., 1.)
        edit_imgs = 127.5*(edit_imgs + 1).permute(0,2,3,1).cpu().numpy()
        edit_imgs = edit_imgs.astype(np.uint8)
        timesteps = self.pipeline.scheduler.timesteps
        timesteps = torch.cat([timesteps] * batch_size)
        if accelerator is not None:
            denoising_log_prob, _ = accelerator.unwrap_model(self.actor).forward_actor(
                hidden_feature_list, timesteps, denoising_list, None
            )
            values = accelerator.unwrap_model(self.critic).forward_critic(hidden_feature_list, timesteps)
        else:
            denoising_log_prob, _ = self.actor.forward_actor(
                hidden_feature_list, timesteps, denoising_list, None
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
        reward = reward.to(src_embeddings.device).unsqueeze(-1).to(src_embeddings.device)
        denoising_log_prob = denoising_log_prob.squeeze(-1)
        denoising_advantage = reward - values
        return Experience(
            x0,
            denoising_list,
            hidden_feature_list,
            denoising_log_prob,
            reward,
            denoising_advantage,
            None,
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
        self.pipeline.unet = UnetDownSampling.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
        self.pipeline.scheduler = DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
        self.pipeline.scheduler.set_timesteps(self.num_inference_steps)

        self.actor = UnetActorDiscrete(1280*2, 768, 2, 2)
        self.critic = UnetCriticDiscrete(1280*2, 768)

        self.pipeline.vae.requires_grad_(False)
        self.pipeline.unet.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)

        self.pipeline.to(accelerator.device)
        self.pipeline.vae.to(accelerator.device)
        self.pipeline.unet.to(accelerator.device)
        self.pipeline.text_encoder.to(accelerator.device)

        self.actor.to(accelerator.device)
        self.critic.to(accelerator.device)

        accelerator.wait_for_everyone()

        actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                           lr=self.actor_lr,
                                           betas=(self.adam_beta1, self.adam_beta2))
        critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                            lr=self.critic_lr,
                                            betas=(self.adam_beta1, self.adam_beta2))
        dataloader = DataLoader(
            self.dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn=collate_fn
        )

        self.actor, self.critic, actor_optimizer, critic_optimizer, dataloader = accelerator.prepare(
            self.actor, self.critic, actor_optimizer, critic_optimizer, dataloader
        )
        print(f"Finish preparing")
        global_step = 0
        total_train_steps = self.args.num_epochs * len(dataloader)
        progress_bar = tqdm(range(global_step, total_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")
        for epoch in range(self.args.num_epochs):
            for step, batch in enumerate(dataloader):
                with accelerator.accumulate(self.actor):
                    src_input_ids = batch["src_input_ids"].to(accelerator.device)
                    tgt_input_ids = batch["tgt_input_ids"].to(accelerator.device)
                    src_embeddings = self.pipeline.text_encoder(src_input_ids)[0]
                    tgt_embeddings = self.pipeline.text_encoder(tgt_input_ids)[0]
                    images = batch["images"].to(accelerator.device)
                    editing_region_masks = batch["masks"].to(accelerator.device)
                    edit_prompts = batch["edit_prompts"]
                    experience = self.make_experience(
                        images,
                        src_embeddings,
                        tgt_embeddings,
                        editing_region_masks,
                        edit_prompts,
                        accelerator
                    )
                    self.actor.train()

                    hidden_feature_list = experience.hidden_feature_list
                    batch_size, _ = hidden_feature_list.shape[:2]
                    timesteps = self.pipeline.scheduler.timesteps
                    timesteps = torch.cat([timesteps]*batch_size)

                    denoising_actor_log_probs, _ = accelerator.unwrap_model(self.actor).forward_actor(
                        experience.hidden_feature_list,
                        timesteps,
                        experience.denoising_list,
                        None
                    )
                    denoising_actor_log_probs = denoising_actor_log_probs.squeeze(-1)
                    actor_loss = self.actor_loss(
                        denoising_actor_log_probs,
                        experience.denoising_log_prob,
                        experience.denoising_advantage
                    )
                    params_to_clip = list(self.actor.parameters())
                    accelerator.backward(actor_loss)
                    accelerator.clip_grad_norm_(params_to_clip, 1.0)
                    actor_optimizer.step()
                    actor_optimizer.zero_grad()

                    self.critic.train()
                    new_values = accelerator.unwrap_model(self.critic).forward_critic(
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
                            "critic_loss": critic_loss.detach().item(),
                            "rewards": experience.rewards.mean().item(),
                        }
                        progress_bar.update(1)
                        progress_bar.set_postfix(**logs)
                        writer.add_scalar("actor_loss", actor_loss.detach().item(), global_step)
                        writer.add_scalar("critic_loss", critic_loss.detach().item(), global_step)
            if epoch % self.args.checkpointing_steps == 0:
                save_path = os.path.join(self.args.exp_name, f"checkpoint_{epoch}.ckpt")
                self.save_checkpoint(save_path, accelerator)