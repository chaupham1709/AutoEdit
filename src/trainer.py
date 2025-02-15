import torch
from .loss import PolicyLoss, ValueLoss
from .inversion_utils import inversion_forward_process, encode_text, reverse_step
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.nn import functional as F
from torch.cuda.amp.grad_scaler import GradScaler
from tqdm import tqdm
from typing import Union
from dataclasses import dataclass

# Define the reward function
class RewardFunction:
    def __init__(self):
        pass
    def forward(self, x0, src_prompt, tgt_prompt, masks):
        # Need to rewrite this one
        batch_size = x0.shape[0]
        return torch.ones([batch_size, 1]).to(x0.device)

@dataclass
class Experience:
    x0: torch.Tensor
    latent_lists: torch.Tensor
    action_lists: torch.Tensor
    hidden_features_list: torch.Tensor
    actor_log_probs: torch.Tensor
    kl_penalized_reward: torch.Tensor
    advantage: torch.Tensor
    estimated_kl: torch.Tensor,
    values: torch.Tensor

class PPOTrainerDDPMInv:
    def __init__(self,
                cfg,
                actor,
                critic,
                sft_model,
                train_dataset,
                pipeline,
                device):
        
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.run_name = f"ppo_{cfg.exp_name}_{datetime.now().strftime('%m%d%H')}"

        self.orig_actor = actor
        self.orig_critic = critic
        self.orig_sft_model = sft_model

        self.actor = torch.compile(self.orig_actor)
        self.critic = torch.compile(self.orig_critic)
        self.sft_model = torch.compile(self.orig_sft_model)

        self.scorer = RewardFunction()
        self.actor_criterion = PolicyLoss()
        self.critic_criterion = ValueLoss()

        self.train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr = cfg.actor_lr,
            betas=(self.cfg.adam_beta1, self.cfg.adam_beta1)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=cfg.critic_lr,
            betas=(self.cfg.adam_beta1, self.cfg.adam_beta2)
        )

        self.writer = SummaryWriter(f"./runs/{self.run_name}/logs", max_queue=50)
        self.total_epochs = cfg.total_epochs
        self.debug = False
        self.save_freq = 1000
        self.dtype = torch.float16
        self.pipeline = pipeline
        self.step = 0
    
    @torch.no_grad()
    def batch_generate(self,
                       image,
                       src_prompt,
                       tgt_prompt,
                       temperature=1.0,
                       etas = 0):
        w0 = (self.pipeline.vae.encode(image).latent_dist.mode() * 0.18215).float()
        # Find zS based on ddpm inversion
        wt, zs, wts = inversion_forward_process(self.pipeline, w0, etas=etas, prompt=src_prompt, cfg_scale=self.cfg.scale_src, num_inference_steps=self.pipeline.scheduler.num_inference_steps)
        
        batch_size = len(src_prompt)
        cfg_scales_tensor = torch.Tensor([self.cfg.scale_tar]).float().view(-1,1,1,1).to(self.pipeline.device)
        uncond_embedding = encode_text(self.pipeline, [""]*batch_size)

        if etas is None: etas = 0
        if type(etas) in [int, float]: etas = [etas]*self.pipeline.scheduler.num_inference_steps
        assert len(etas) == self.pipeline.scheduler.num_inference_steps
        timesteps = self.pipeline.scheduler.timesteps.to(self.pipeline.device)
        xT = wts[:, len(self.pipeline.scheduler.timesteps)]
        xt = xT
        action_lists = []
        latent_lists = []
        hidden_features_list = []
        for t in timesteps:
            idx = len(self.pipeline.scheduler.timesteps) - 1
            hidden_features, uncond_out = self.pipeline.unet(xt, t, encoder_hidden_states=uncond_embedding)
            logits = self.actor(hidden_features)
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_action = torch.multinomial(probs, num_samples=1)
            latent_lists.append(xt)
            hidden_features_list.append(hidden_features)
            action_lists.append(next_action)
            prompt_embeds = tgt_prompt * next_action + src_prompt * (1-next_action)
            _, cond_out = self.pipeline.unet(xt, t, encoder_hidden_states=prompt_embeds)

            # Classifier free guidance
            noise_pred = uncond_out.sample + cfg_scales_tensor * (cond_out.sample - uncond_out.sample)
            z_latent = zs[:,idx]
            xt = reverse_step(self.pipeline, noise_pred, t, xt, eta=etas[idx], variance_noise=z_latent)
        
        action_lists = torch.cat(action_lists, dim=-1).long()
        latent_lists = torch.stack(latent_lists, dim=1)
        hidden_features_list = torch.stack(hidden_features_list, dim=1)
        return xt, latent_lists, action_lists, hidden_features_list
    
    @torch.no_grad()
    def make_experience(self, image, src_prompt, tgt_prompt):
        self.sft_model.eval()
        self.actor.eval()
        self.critic.eval()
        x0, latent_lists, action_lists, hidden_features_list = self.batch_generate(image, src_prompt, tgt_prompt, etas=1.)
        actor_log_probs = self.actor.forward_actor(
            hidden_features_list, action_lists
        )
        sft_log_probs = self.sft_model.forward_actor(
            hidden_features_list, action_lists
        )
        values = self.critic.forward_critic(hidden_features_list)
        reward = self.scorer.get_score_batch(x0, src_prompt, tgt_prompt, None)
        kl_penalized_reward, estimated_kl = self.kl_penalized_reward(
            reward, actor_log_probs, sft_log_probs
        )
        advantage = kl_penalized_reward - values
        return Experience(
            x0,
            latent_lists,
            action_lists,
            hidden_features_list,
            actor_log_probs,
            kl_penalized_reward,
            advantage,
            estimated_kl
        )
    
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

    
    def fit(self):
        scaler = GradScaler(enabled=self.dtype!=torch.float32)
        for epoch in range(self.total_epochs):
            for step, batch in enumerate(
                pbar := tqdm(self.train_dataloader)
            ):
                step = step + self.step
                src_input_ids = batch["src_input_ids"].to(self.device)
                tgt_input_ids = batch["tgt_input_ids"].to(self.device)
                image = batch["image"].to(self.device)
                mask = batch["mask"].to(self.device)
                total_steps = step + epoch * len(self.train_dataloader)

                with torch.autocast(
                    device_type="cuda",
                    dtype=self.dtype,
                    enabled=self.dtype != torch.float32
                ):
                    src_embeddings = self.pipeline.text_encoder(src_input_ids)[0]
                    tgt_embeddings = self.pipeline.text_encoder(tgt_input_ids)[0]
                    experience = self.make_experience(image, src_embeddings, tgt_embeddings)
                    self.actor.train()
                    curr_actor_log_probs = self.actor.forward_actor(
                        experience.hidden_features_list, experience.action_lists
                    )
                    actor_loss = self.actor_criterion(
                        curr_actor_log_probs,
                        experience.actor_log_probs,
                        experience.advantage
                    )
                    scaler.scale(actor_loss).backward()
                    scaler.step(self.actor_optimizer)
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    actor_lossf = actor_loss.item()

                    self.critic.train()
                    new_values = self.critic.forward_critic(
                        experience.completion,
                        experience.attention_mask,
                        experience.num_actions,
                    ).view(-1, 1)

                    critic_loss = self.critic_criterion(
                        new_values,
                        experience.kl_penalized_reward,
                        experience.values,
                        experience.action_mask,
                    )

                    scaler.scale(critic_loss).backward()
                    scaler.step(self.critic_optimizer)
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    critic_lossf = critic_loss.item()

                    scaler.update()
