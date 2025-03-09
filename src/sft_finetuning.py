from torch.utils.data import DataLoader
from .dataset import collate_fn
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
import os
import torch
from tqdm import tqdm
from .inversion_utils import encode_text
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
class SFTFineTuningHead:
    def __init__(self,
                args,
                cfg,
                model,
                train_dataset,
                pipeline,
                inv_scheduler
                ):
        self.cfg = cfg
        self.model = model
        self.train_dataloader = DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn, shuffle=False)
        self.num_epochs = 20
        self.pipeline = pipeline
        self.args = args
        self.inv_scheduler = inv_scheduler
        self.inv_scheduler.set_timesteps(50)
        self.pipeline.scheduler.set_timesteps(50)
        self.criterion = CrossEntropyLoss()
    
    @torch.no_grad()
    def invert_image_to_noise(self, images, src_embeddings):
        latents = (self.pipeline.vae.encode(images).latent_dist.sample() * 0.18215).float()
        for t in self.inv_scheduler.timesteps:
            noise_pred = self.pipeline.unet(latents, t, src_embeddings).sample
            latents = self.inv_scheduler.step(noise_pred, t, latents).prev_sample
        
        return latents
    
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
    
    def save_checkpoint(self, ckpt_name):
        torch.save(self.model.state_dict(), ckpt_name)

    def fit(self):
        # Setup for accelerate
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

        self.pipeline.vae.to(accelerator.device)
        self.pipeline.unet.to(accelerator.device)
        self.pipeline.text_encoder.to(accelerator.device)
        self.model.to(accelerator.device)

        model_optimizer = torch.optim.Adam(self.model.parameters(),
                                           lr=0.5e-3)
        
        self.model, model_optimizer, self.train_dataloader = accelerator.prepare(
            self.model, model_optimizer, self.train_dataloader
        )
        global_step = 0
        total_train_steps = self.num_epochs * len(self.train_dataloader)
        progress_bar = tqdm(range(global_step, total_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")
        for epoch in range(self.num_epochs):
            for step, batch in enumerate(self.train_dataloader):
                self.model.train()
                with accelerator.accumulate(self.model):
                    src_input_ids = batch["src_input_ids"].to(accelerator.device)
                    tgt_input_ids = batch["tgt_input_ids"].to(accelerator.device)
                    images = batch["images"].to(accelerator.device)

                    src_outputs = self.pipeline.text_encoder(src_input_ids)
                    src_embeddings, src_pooled_embeddings = src_outputs[0], src_outputs[1]
                    tgt_outputs = self.pipeline.text_encoder(tgt_input_ids)
                    tgt_embeddings, tgt_pooled_embeddings = tgt_outputs[0], tgt_outputs[1]
                    latents = self.invert_image_to_noise(images, src_embeddings)
                    batch_size = latents.shape[0]
                    ratio = 0.2+ 0.4*torch.rand((batch_size,))
                    trajectories = self.make_trajectory(batch_size, ratio, num_timesteps=50).to(latents.device)
                    _, hidden_features_lists, latent_lists = self.denoising_trajectory(latents, trajectories, src_embeddings, tgt_embeddings)
                    batch_size, num_timesteps, channels, height, width = hidden_features_lists.shape

                    hidden_features_lists = hidden_features_lists.view(num_timesteps * batch_size, channels, height, width)
                    src_pooled_embeddings = torch.stack([src_pooled_embeddings] * num_timesteps, dim=1).view(num_timesteps*batch_size, -1)
                    tgt_pooled_embeddings = torch.stack([tgt_pooled_embeddings] * num_timesteps, dim=1).view(num_timesteps*batch_size, -1)
                    
                    logits = self.model(hidden_features_lists, src_pooled_embeddings, tgt_pooled_embeddings)
                    trajectories = trajectories.view(-1)
                    loss = self.criterion(logits, trajectories)
                    
                    accelerator.backward(loss)
                    model_optimizer.step()
                    model_optimizer.zero_grad()
                    if accelerator.is_main_process:
                        global_step += 1
                        logs = {"Epoch": epoch, "loss": loss.detach().item()}
                        progress_bar.update(1)
                        progress_bar.set_postfix(**logs)
                        writer.add_scalar("Loss/loss", loss.item(), global_step)

            if epoch % self.args.checkpointing_steps == 0:
                save_path = os.path.join(self.args.exp_name, f"checkpoint_{epoch}.ckpt")
                self.save_checkpoint(save_path)
            
        if accelerator.is_main_process:
            writer.close()
    
    @torch.no_grad()
    def test_generation_trajectory(self):
        self.pipeline.to("cuda")
        for batch in self.train_dataloader:
            src_input_ids = batch["src_input_ids"].to("cuda")
            tgt_input_ids = batch["tgt_input_ids"].to("cuda")
            images = batch["images"].to("cuda")

            src_outputs = self.pipeline.text_encoder(src_input_ids)
            src_embeddings, src_pooled_embeddings = src_outputs[0], src_outputs[1]
            tgt_outputs = self.pipeline.text_encoder(tgt_input_ids)
            tgt_embeddings, tgt_pooled_embeddings = tgt_outputs[0], tgt_outputs[1]
            latents = self.invert_image_to_noise(images, src_embeddings)
            batch_size = latents.shape[0]
            ratio = torch.tensor([1.0]*batch_size)
            trajectories = self.make_trajectory(batch_size, ratio, num_timesteps=50).to(latents.device)
            print(trajectories)
            latents, _, _ = self.denoising_trajectory(latents, trajectories, src_embeddings, tgt_embeddings)
            images = self.pipeline.vae.decode(latents / 0.18215).sample
            images = torch.clamp(images, -1, 1).float()
            import numpy as np
            images = 127.5 * (images + 1).cpu().permute(0,2,3,1).numpy()
            images = images.astype(np.uint8)
            from PIL import Image
            Image.fromarray(images[1]).save("denoising_results.png")
            break
    
    @torch.no_grad()
    def test_sft_finetuning_phase(self):
        checkpoint_path = os.path.join(self.args.exp_name, "checkpoint_19.ckpt")
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.to("cuda")
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
        latents = self.invert_image_to_noise(images, src_embeddings)
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
            logits = self.model(hidden_features, src_pooled_embeddings, tgt_pooled_embeddings)
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
