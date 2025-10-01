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
from actor_critic import UnetDownSampling, UnetActorDiscrete, UnetCriticDiscrete
from ddpm_p2p_wo_attention import Prompt2PromptDDPMPPO
from prompt_to_prompt.ptp_classes import load_512

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

class DDPMPrompt2PromptCFG(Prompt2PromptDDPMPPO):
    def __init__(self,
                 args,
                 dataset,
                 device):
        super().__init__(args, dataset, device, alpha1=3.0, alpha2=1.0)
        self.guidance_scale_set = [4.5, 5.5, 6.0, 7.5, 8.0, 9.0]
        self.skip_step = 15
        self.cfg_scale_src = 3.0
        self.eta = 1.0
    
    def init_model(self):
        model_id = "CompVis/stable-diffusion-v1-4"
        self.pipeline = StableDiffusionPipeline.from_pretrained(model_id).to(self.device)
        self.pipeline.unet = UnetDownSampling.from_pretrained(model_id, subfolder="unet").to(self.device)
        self.pipeline.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.pipeline.scheduler.set_timesteps(self.num_inference_steps)
        self.actor = UnetActorDiscrete(1280*2, 768, len(self.guidance_scale_set), 2)
        if self.args.checkpoint:
            state_dict = torch.load(self.args.checkpoint, map_location=self.device)
            if "actor" in state_dict:
                self.actor.load_state_dict(state_dict["actor"])
            else:
                self.actor.load_state_dict(state_dict)
    
    @torch.no_grad()
    def batch_generate(self,
                       images,
                       src_prompt_embeddings,
                       tgt_prompt_embeddings,
                       temperature=1.0):
        batch_size = images.shape[0]
        latents = (self.pipeline.vae.encode(images).latent_dist.mode() * 0.18215)
        uncond_embeddings = encode_text(self.pipeline, [""] * batch_size)
        wt, zs, xts = inversion_forward_process(
            self.pipeline,
            latents,
            src_prompt_embeddings,
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
            hidden_features_src, src_cond_out = self.pipeline.unet.forward(
                xt,
                t,
                encoder_hidden_states=src_prompt_embeddings,
                return_hidden=True
            )
            src_cond_out = src_cond_out.sample
            uncond_out = self.pipeline.unet(
                xt,
                t,
                encoder_hidden_states=uncond_embeddings
            ).sample
            src_noise_pred = uncond_out + self.cfg_scale_src * (src_cond_out - uncond_out)

            if i > self.skip_step:
                hidden_features_tgt, tgt_cond_out = self.pipeline.unet.forward(
                    xt,
                    t,
                    encoder_hidden_states=tgt_prompt_embeddings,
                    return_hidden=True
                )
                tgt_cond_out = tgt_cond_out.sample

                hidden_features = torch.cat([hidden_features_src, hidden_features_tgt], dim=1)
                hidden_feature_list.append(hidden_features)
                timesteps = torch.cat([t.unsqueeze(0)]*batch_size)
                logits, _ = self.actor(hidden_features, timesteps)
                probs = F.softmax(logits / temperature, dim=-1)
                next_denoise_action = torch.multinomial(probs, num_samples=1)

                action_list.append(next_denoise_action)
                cfg_scale_tgt = [self.guidance_scale_set[a.item()] for a in next_denoise_action]
                cfg_scale_tgt = torch.tensor(cfg_scale_tgt, device=xt.device)[:, None, None, None]
                tgt_noise_pred = uncond_out + cfg_scale_tgt * (tgt_cond_out - uncond_out)
                noise_pred = tgt_noise_pred
            else:
                noise_pred = src_noise_pred
            idx = self.num_inference_steps - t_to_idx[int(t)] - 1
            z = zs[:, idx, :, :, :]
            xt = reverse_step(
                self.pipeline,
                noise_pred,
                t, xt,self.eta, z
            )
        hidden_feature_list = torch.stack(hidden_feature_list, dim=1)
        action_list = torch.cat(action_list, dim=1)
        return xt, hidden_feature_list, action_list
    
    @torch.no_grad()
    def make_experience(self,
                        images,
                        src_prompt_embeddings,
                        tgt_prompt_embeddings,
                        editing_region_masks,
                        edit_prompts,
                        accelerator=None):
        self.actor.eval()
        self.critic.eval()
        x0, hidden_feature_list, denoising_list = self.batch_generate(
            images,
            src_prompt_embeddings,
            tgt_prompt_embeddings
        )
        batch_size = hidden_feature_list.shape[0]
        edit_imgs = self.pipeline.vae.decode(x0/0.18215).sample
        edit_imgs = torch.clamp(edit_imgs, -1., 1.)
        edit_imgs = 127.5*(edit_imgs + 1).permute(0,2,3,1).cpu().numpy()
        edit_imgs = edit_imgs.astype(np.uint8)
        timesteps = self.pipeline.scheduler.timesteps[self.skip_step+1:]
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
        reward = reward.to(src_prompt_embeddings.device).unsqueeze(-1).to(src_prompt_embeddings.device)
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

        self.actor = UnetActorDiscrete(1280*2, 768, len(self.guidance_scale_set), 2)
        self.critic = UnetCriticDiscrete(1280*2, 768)

        self.pipeline.to(accelerator.device)

        self.pipeline.vae.requires_grad_(False)
        self.pipeline.unet.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)

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
                    timesteps = self.pipeline.scheduler.timesteps[self.skip_step+1:]
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
    
    @torch.no_grad()
    def image_generation(self,
                         image_path,
                         src_prompt,
                         tgt_prompt,
                         prog_bar=True,
                         run_baseline=False,
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
            etas=1.0,
            cfg_scale=self.cfg_scale_src,
            num_inference_steps=self.num_inference_steps
        )
        timesteps = self.pipeline.scheduler.timesteps.to(self.device)
        xt = xts[:, -1, :, :, :]
        t_to_idx = {int(v):k for k,v in enumerate(timesteps)}
        op = tqdm(timesteps) if prog_bar else timesteps
        all_cfg_scale_tgt = []
        for i, t in enumerate(op):
            hidden_features_src, src_cond_out = self.pipeline.unet.forward(
                xt,
                t,
                encoder_hidden_states=src_prompt_embedding,
                return_hidden=True
            )
            src_cond_out = src_cond_out.sample
            uncond_out = self.pipeline.unet(
                xt,
                t,
                encoder_hidden_states=uncond_embeddings
            ).sample
            src_noise_pred = uncond_out + self.cfg_scale_src * (src_cond_out - uncond_out)

            if i > self.skip_step:
                hidden_features_tgt, tgt_cond_out = self.pipeline.unet.forward(
                    xt,
                    t,
                    encoder_hidden_states=tgt_prompt_embedding,
                    return_hidden=True
                )
                tgt_cond_out = tgt_cond_out.sample

                hidden_features = torch.cat([hidden_features_src, hidden_features_tgt], dim=1)
                timesteps = torch.cat([t.unsqueeze(0)] * batch_size)
                logits, _ = self.actor(hidden_features, timesteps)
                probs = F.softmax(logits / temperature, dim=-1)
                action = torch.multinomial(probs, num_samples=1)
                if not run_baseline:
                    cfg_scale_tgt = [self.guidance_scale_set[a.item()] for a in action]
                    cfg_scale_tgt = torch.tensor(cfg_scale_tgt, device=xt.device)[:, None, None, None]
                else:
                    cfg_scale_tgt = 7.5
                all_cfg_scale_tgt.append(cfg_scale_tgt.item())
                tgt_noise_pred = uncond_out + cfg_scale_tgt * (tgt_cond_out - uncond_out)
                noise_pred = tgt_noise_pred
            else:
                noise_pred = src_noise_pred
            idx = self.num_inference_steps - t_to_idx[int(t)] - 1
            z = zs[:, idx, :, :, :]
            xt = reverse_step(
                self.pipeline,
                noise_pred,
                t, xt, self.eta, z
            )
        
        with autocast("cuda"), inference_mode():
            x0_dec = self.pipeline.vae.decode(1 / 0.18215 * xt).sample
        
        img = image_grid(x0_dec)
        return img, all_cfg_scale_tgt
