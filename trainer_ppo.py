import argparse
from src.config import get_configs
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler
from src.actor_critic import UnetActorDiscrete, UnetDownSampling, UnetCriticDiscrete
from src.dataset import AutoEditDataset
from src.trainer_ddim import PPOTrainerDDIM
import torch
import numpy as np
from PIL import Image

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mode", type=str, default="ddpm_inv")
    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    parser.add_argument("--config_scale_src", type=float,default=3.5)
    parser.add_argument("--config_scale_tar", type=float, default=15)
    parser.add_argument("--exp_name", type=str, default="ddpm_inv")
    parser.add_argument("--logging_dir", type=str, default="log")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--checkpointing_steps", type=int, default=1)
    parser.add_argument("--checkpoint_path", type=str)
    args = parser.parse_args()
    return args

def main(args):
    cfg = get_configs("gpt2-medium")
    cfg.scale_src = args.config_scale_src
    cfg.scale_tar = args.config_scale_tar
    cfg.exp_name = args.exp_name
    model_id = "CompVis/stable-diffusion-v1-4"
    pipeline = StableDiffusionPipeline.from_pretrained(model_id)
    del pipeline.unet

    unet = UnetDownSampling.from_pretrained(model_id, subfolder="unet")
    pipeline.unet = unet
    inv_scheduler = DDIMInverseScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipeline.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipeline.to("cuda")
    actor = UnetActorDiscrete(1280*2, 768)
    sft_model = UnetActorDiscrete(1280*2, 768)
    if args.checkpoint_path:
        actor.load_state_dict(torch.load(args.checkpoint_path))
        sft_model.load_state_dict(torch.load(args.checkpoint_path))
    
    train_dataset = AutoEditDataset(args.dataset_dir, pipeline.tokenizer)
    critic = UnetCriticDiscrete(1280*2, 768)
    trainer = PPOTrainerDDIM(args, cfg, actor, critic, sft_model, train_dataset, pipeline, inv_scheduler, device="cuda")
    # trainer.fit()
    trainer.test_trainer_ddim()
    

if __name__ == "__main__":
    args = parse_arguments()
    main(args)