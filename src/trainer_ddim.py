import torch
from .loss import PolicyLoss, ValueLoss
from .dataset import collate_fn
from torch.utils.data import DataLoader
from .inversion_utils import encode_text
import numpy as np
from torch.nn import functional as F
from .reward import RewardFunction
from typing import Union
from dataclasses import dataclass
import os
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

@dataclass
class Experience:
    x0: torch.Tensor
    latent_lists: torch.Tensor
    action_lists: torch.Tensor
    hidden_features_list: torch.Tensor
    actor_log_probs: torch.Tensor
    kl_penalized_reward: torch.Tensor
    advantage: torch.Tensor
    estimated_kl: torch.Tensor
    values: torch.Tensor

class PPOTrainerDDIM:
    def __init__(self,
                 args,
                 cfg,
                 actor,
                 critic,
                 sft_model,
                 train_dataset,
                 pipeline,
                 inv_scheduler,
                 device):
        self.args = args
        self.cfg = cfg
        self.actor = actor
        self.critic = critic
        self.sft_model = sft_model
        
        self.actor_criterion = PolicyLoss()
        self.critic_criterion = ValueLoss()
        self.total_epochs = cfg.total_epochs
        self.pipeline = pipeline
        self.inv_scheduler = inv_scheduler
        self.device = device

        self.pipeline.scheduler.set_timesteps(50)
        self.inv_scheduler.set_timesteps(50)
        self.scorer = RewardFunction()
        self.num_epochs = 30
        self.train_dataloader = DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn, shuffle=False)
    
    @torch.no_grad()
    def invert_image_to_noise(self, images, src_input_ids):
        src_embeddings = self.pipeline.text_encoder(src_input_ids.to(self.device))[0]
        latents = (self.pipeline.vae.encode(images).latent_dist.sample() * 0.18215).float()
        for t in self.inv_scheduler.timesteps:
            noise_pred = self.pipeline.unet(latents, t, src_embeddings).sample
            latents = self.inv_scheduler.step(noise_pred, t, latents).prev_sample
        return latents
    
    @torch.no_grad()
    def denoise_noise_to_image(self, latents, src_input_ids, tgt_input_ids, skip_timestep):
        src_embeddings = self.pipeline.text_encoder(src_input_ids.to(self.device))[0]
        tgt_embeddings = self.pipeline.text_encoder(tgt_input_ids.to(self.device))[0]
        uncond_embeddings = encode_text(self.pipeline, "")
        batch_size = latents.shape[0]
        uncond_embeddings = torch.cat([uncond_embeddings] * batch_size)
        for i, t in enumerate(self.pipeline.scheduler.timesteps):
            if i < skip_timestep:
                noise_pred = self.pipeline.unet(latents, t, src_embeddings).sample
                latents = self.pipeline.scheduler.step(noise_pred, t, latents).prev_sample
            else:
                noise_pred = self.pipeline.unet(latents, t, tgt_embeddings).sample
                uncond_noise_pred = self.pipeline.unet(latents, t, uncond_embeddings).sample
                noise_pred = uncond_noise_pred + 7.5 * (noise_pred - uncond_noise_pred)
                latents = self.pipeline.scheduler.step(noise_pred, t, latents).prev_sample
        
        image = self.pipeline.vae.decode(latents / 0.18215).sample
        image = torch.clamp(image, -1, 1).float()
        image = 127.5 * (image + 1).cpu().permute(0,2,3,1).numpy()
        return image.astype(np.uint8)
    
    @torch.no_grad()
    def convert_image_to_origin_space(self, image):
        image = 127.5*(image + 1).permute(0,2,3,1).cpu().numpy()
        return image.astype(np.uint8)
    
    @torch.no_grad()
    def batch_generate(
        self,
        images,
        src_input_ids,
        tgt_input_ids,
        temperature=1.0
    ):
        latents = self.invert_image_to_noise(images, src_input_ids)
        src_outputs = self.pipeline.text_encoder(src_input_ids.to(self.device))
        src_embeddings = src_outputs[0]; src_pooled_embeddings = src_outputs[1]
        tgt_outputs = self.pipeline.text_encoder(tgt_input_ids.to(self.device))
        tgt_embeddings = tgt_outputs[0]; tgt_pooled_embeddings = tgt_outputs[1]
        uncond_embeddings = encode_text(self.pipeline, "")
        batch_size = latents.shape[0]
        uncond_embeddings = torch.cat([uncond_embeddings] * batch_size)
        action_lists = []
        latent_lists = []
        hidden_features_list = []
        for t in self.pipeline.scheduler.timesteps:
            hidden_features_src, src_noise_pred = self.pipeline.unet(latents, t, encoder_hidden_states=src_embeddings, return_hidden=True)
            src_noise_pred = src_noise_pred.sample
            hidden_features_tar, tgt_noise_pred = self.pipeline.unet(latents, t, encoder_hidden_states=tgt_embeddings, return_hidden=True)
            tgt_noise_pred = tgt_noise_pred.sample
            uncond_noise_pred = self.pipeline.unet(latents, t, encoder_hidden_states=uncond_embeddings).sample
            hidden_features = torch.cat([hidden_features_src, hidden_features_tar], dim=1)
            logits = self.actor(hidden_features, src_pooled_embeddings, tgt_pooled_embeddings)
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_action = torch.multinomial(probs, num_samples=1)
            latent_lists.append(latents)
            hidden_features_list.append(hidden_features)
            action_lists.append(next_action)

            tgt_noise_pred = uncond_noise_pred + 7.5 * (tgt_noise_pred - uncond_noise_pred)
            masking = (next_action[:,:,None,None] == 0).float()
            noise_pred = src_noise_pred * masking + (1-masking) * tgt_noise_pred
            latents = self.pipeline.scheduler.step(noise_pred, t, latents).prev_sample

        action_lists = torch.cat(action_lists, dim=-1).long()
        latent_lists = torch.stack(latent_lists, dim=1)
        hidden_features_list = torch.stack(hidden_features_list, dim=1)
        return latents, latent_lists, action_lists, hidden_features_list
    
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
        return reward - self.cfg.kl_beta * estimated_kl, estimated_kl
    
    @torch.no_grad()
    def make_experience(self, images, src_input_ids, tgt_input_ids, editing_region_masks, editing_types, changed_words):
        self.sft_model.eval()
        self.actor.eval()
        self.critic.eval()
        x0, latent_lists, action_lists, hidden_feature_list = self.batch_generate(images, src_input_ids, tgt_input_ids)
        src_outputs = self.pipeline.text_encoder(src_input_ids.to(self.device))
        src_embeddings = src_outputs[0]; src_pooled_embeddings = src_outputs[1]
        tgt_outputs = self.pipeline.text_encoder(tgt_input_ids.to(self.device))
        tgt_embeddings = tgt_outputs[0]; tgt_pooled_embeddings = tgt_outputs[1]
        batch_size, num_timesteps = hidden_feature_list.shape[:2]
        src_pooled_embeddings = torch.stack([src_pooled_embeddings] * num_timesteps, dim=1).view(batch_size * num_timesteps, -1)
        tgt_pooled_embeddings = torch.stack([tgt_pooled_embeddings] * num_timesteps, dim=1).view(batch_size * num_timesteps, -1)
        
        edit_imgs = self.pipeline.vae.decode(x0 / 0.18215).sample
        edit_imgs = torch.clamp(edit_imgs, -1., 1.)
        edit_imgs = 127.5*(edit_imgs + 1).permute(0,2,3,1).cpu().numpy()
        edit_imgs = edit_imgs.astype(np.uint8)
        actor_log_probs = self.actor.forward_actor(
            hidden_feature_list, src_pooled_embeddings, tgt_pooled_embeddings, action_lists
        )
        sft_log_probs = self.sft_model.forward_actor(
            hidden_feature_list, src_pooled_embeddings, tgt_pooled_embeddings, action_lists
        )
        values = self.critic.forward_critic(hidden_feature_list)
        origin_images = self.convert_image_to_origin_space(images)
        reward = self.scorer.estimate_reward(
            origin_images,
            editing_region_masks.to("cpu"),
            edit_imgs,
            editing_types.to("cpu"),
            changed_words
        )
        reward = reward.to(src_input_ids.device).unsqueeze(-1)
        actor_log_probs = actor_log_probs.squeeze(-1)
        sft_log_probs = sft_log_probs.squeeze(-1)
        kl_penalized_reward, estimate_kl = self.kl_penalized_reward(
            reward,
            actor_log_probs,
            sft_log_probs
        )
        advantage = kl_penalized_reward - values
        return Experience(
            x0,
            latent_lists,
            action_lists,
            hidden_feature_list,
            actor_log_probs,
            kl_penalized_reward,
            advantage,
            estimate_kl,
            values
        )
    
    def save_checkpoint(self, checkpoint_path):
        actor_state_dict = self.actor.state_dict()
        critic_state_dict = self.critic.state_dict()
        model = {
            "actor": actor_state_dict,
            "critic": critic_state_dict
        }
        torch.save(model, checkpoint_path)
    
    def fit(self):
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
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.unet.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)
        self.sft_model.requires_grad_(False)

        self.pipeline.vae.to(accelerator.device)
        self.pipeline.unet.to(accelerator.device)
        self.pipeline.text_encoder.to(accelerator.device)

        self.actor.to(accelerator.device)
        self.critic.to(accelerator.device)
        self.sft_model.to(accelerator.device)

        actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                           lr=self.cfg.actor_lr,
                                           betas=(self.cfg.adam_beta1, self.cfg.adam_beta2))
        critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                            lr=self.cfg.critic_lr,
                                            betas=(self.cfg.adam_beta1, self.cfg.adam_beta2))
        self.actor, self.critic, actor_optimizer, critic_optimizer, self.train_dataloader = accelerator.prepare(
            self.actor, self.critic, actor_optimizer, critic_optimizer, self.train_dataloader
        )
        global_step = 0
        total_train_steps = self.num_epochs * len(self.train_dataloader)
        progress_bar = tqdm(range(global_step, total_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")
        for epoch in range(self.num_epochs):
            for step, batch in enumerate(self.train_dataloader):
                with accelerator.accumulate(self.actor):
                    src_input_ids = batch["src_input_ids"].to(accelerator.device)
                    tgt_input_ids = batch["tgt_input_ids"].to(accelerator.device)
                    images = batch["images"].to(accelerator.device)

                    src_outputs = self.pipeline.text_encoder(src_input_ids)
                    src_embeddings, src_pooled_embeddings = src_outputs[0], src_outputs[1]
                    tgt_outputs = self.pipeline.text_encoder(tgt_input_ids)
                    tgt_embeddings, tgt_pooled_embeddings = tgt_outputs[0], tgt_outputs[1]
                    editing_region_masks = batch["masks"].to(accelerator.device)
                    editing_type_ids = batch["editing_type_ids"].long().to(accelerator.device)
                    edit_prompts = batch["edit_prompts"]
                    experience = self.make_experience(
                        images, src_input_ids, tgt_input_ids, editing_region_masks, editing_type_ids, edit_prompts
                    )
                    self.actor.train()
                    hidden_features_list = experience.hidden_features_list
                    batch_size, num_timesteps = hidden_features_list.shape[:2]
                    src_pooled_embeddings = torch.stack([src_pooled_embeddings] * num_timesteps, dim=1).view(batch_size*num_timesteps, -1)
                    tgt_pooled_embeddings = torch.stack([tgt_pooled_embeddings] * num_timesteps, dim=1).view(batch_size*num_timesteps, -1)

                    curr_actor_log_probs = self.actor.forward_actor(experience.hidden_features_list,
                                                                    src_pooled_embeddings,
                                                                    tgt_pooled_embeddings,
                                                                    experience.action_lists)
                    curr_actor_log_probs = curr_actor_log_probs.squeeze(-1)
                    actor_loss = self.actor_criterion(
                        curr_actor_log_probs,
                        experience.actor_log_probs,
                        experience.advantage
                    )
                    params_to_clip = list(self.actor.parameters())
                    accelerator.backward(actor_loss)
                    accelerator.clip_grad_norm_(params_to_clip, 1.0)
                    actor_optimizer.step()
                    actor_optimizer.zero_grad()

                    self.critic.train()
                    new_values = self.critic.forward_critic(
                        experience.hidden_features_list
                    ).view(-1,1)
                    critic_loss = self.critic_criterion(
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
                        logs = {"Epoch": epoch, "actor_loss": actor_loss.detach().item(), "critic_loss": critic_loss.detach().item()}
                        progress_bar.update(1)
                        progress_bar.set_postfix(**logs)
                        writer.add_scalar("actor_loss", actor_loss.detach().item(), global_step)
                        writer.add_scalar("critic_loss", critic_loss.detach().item(), global_step)
                
            if epoch % self.args.checkpointing_steps == 0:
                save_path = os.path.join(self.args.exp_name, f"checkpoint_{epoch}.ckpt")
                self.save_checkpoint(save_path)
            
        if accelerator.is_main_process:
            writer.close()
    
    def make_trajectory(self, batch_size, ratios, num_timesteps):
        timestep_actions = torch.zeros((batch_size, num_timesteps)).long()
        num_src_actions = [int(ratio.item() * num_timesteps) for ratio in ratios]
        for i, num_src_action in enumerate(num_src_actions):
            timestep_actions[i, num_src_action:] = 1
        return timestep_actions
    
    @torch.no_grad()
    def denoising_trajectory(self, latents, trajectories, src_embeddings, tgt_embeddings):
        latent_lists = []
        hidden_feature_list = []
        uncond_embeddings = encode_text(self.pipeline, "")
        batch_size = latents.shape[0]
        uncond_embeddings = torch.cat([uncond_embeddings] * batch_size)
        for idx, t in enumerate(self.pipeline.scheduler.timesteps):
            hidden_features_src, src_noise_pred = self.pipeline.unet(latents, t, src_embeddings, return_hidden=True)
            src_noise_pred = src_noise_pred.sample
            uncond_noise_pred = self.pipeline.unet(latents, t, uncond_embeddings).sample
            hidden_features_tgt, tgt_noise_pred = self.pipeline.unet(latents, t, tgt_embeddings, return_hidden=True)
            tgt_noise_pred = tgt_noise_pred.sample
            hidden_features = torch.cat([hidden_features_src, hidden_features_tgt], dim=1)
            hidden_feature_list.append(hidden_features)
            latent_lists.append(latents)

            # if follow tgt embedding, use cfg noise pred
            tgt_noise_pred = uncond_noise_pred + 7.5*(tgt_noise_pred - uncond_noise_pred)
            # masking
            cur_action = trajectories[:,idx]
            masking = (cur_action[:,None,None,None] == 0).float()
            noise_pred = src_noise_pred * masking + (1-masking) * tgt_noise_pred
            latents = self.pipeline.scheduler.step(noise_pred, t, latents).prev_sample
        
        hidden_feature_list = torch.stack(hidden_feature_list, dim=1)
        latent_lists = torch.stack(latent_lists, dim=1)
        return latents, hidden_feature_list, latent_lists
    
    @torch.no_grad()
    def test_trainer_ddim(self):
        checkpoint_path = os.path.join(self.args.exp_name, "checkpoint_5.ckpt")
        self.actor.load_state_dict(torch.load(checkpoint_path)["actor"])
        self.actor.to("cuda")
        for step, batch in enumerate(self.train_dataloader):
            if step==3: break
        src_input_ids = batch["src_input_ids"].to("cuda")
        tgt_input_ids = batch["tgt_input_ids"].to("cuda")
        print(batch["edit_prompts"])
        images = batch["images"].to("cuda")
        src_outputs = self.pipeline.text_encoder(src_input_ids)
        src_embeddings, src_pooled_embeddings = src_outputs[0], src_outputs[1]
        tgt_outputs = self.pipeline.text_encoder(tgt_input_ids)
        tgt_embeddings, tgt_pooled_embeddings = tgt_outputs[0], tgt_outputs[1]
        latents = self.invert_image_to_noise(images, src_input_ids)
        uncond_embeddings = encode_text(self.pipeline, "")
        batch_size = latents.shape[0]
        uncond_embeddings = torch.cat([uncond_embeddings] * batch_size)
        action_lists = []
        for idx, t in enumerate(self.pipeline.scheduler.timesteps):
            hidden_features_src, src_noise_pred = self.pipeline.unet(latents, t, src_embeddings, return_hidden=True)
            src_noise_pred = src_noise_pred.sample
            uncond_noise_pred = self.pipeline.unet(latents, t, uncond_embeddings).sample
            hidden_features_tgt, tgt_noise_pred = self.pipeline.unet(latents, t, tgt_embeddings, return_hidden=True)
            tgt_noise_pred = tgt_noise_pred.sample
            tgt_noise_pred = uncond_noise_pred + 7.5*(tgt_noise_pred - uncond_noise_pred)
            hidden_features = torch.cat([hidden_features_src, hidden_features_tgt], dim=1)
            logits = self.actor(hidden_features, src_pooled_embeddings, tgt_pooled_embeddings)
            cur_action = torch.argmax(logits, dim=1)
            action_lists.append(cur_action)
            masking = (cur_action[:,None,None,None] == 0).float()
            noise_pred = src_noise_pred * masking + (1-masking) * tgt_noise_pred
            latents = self.pipeline.scheduler.step(noise_pred, t, latents).prev_sample
        
        action_lists = torch.stack(action_lists, dim=-1)
        images = self.pipeline.vae.decode(latents / 0.18215).sample
        images = torch.clamp(images, -1, 1).float()
        import numpy as np
        images = 127.5 * (images + 1).cpu().permute(0,2,3,1).numpy()
        images = images.astype(np.uint8)
        from PIL import Image
        Image.fromarray(images[1]).save("denoise_img.png")
        print(action_lists)