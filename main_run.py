import argparse
from src.config import get_configs
from diffusers import StableDiffusionPipeline, DDIMScheduler
from src.actor_critic import UnetDownSampling, UnetActorDiscrete, UnetCriticDiscrete
from src.trainer import PPOTrainerDDPMInv
from src.dataset import AutoEditDataset
import copy

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
    args = parser.parse_args()
    return args

def main(args):
    cfg = get_configs("gpt2-medium")
    cfg.scale_src = args.config_scale_src
    cfg.scale_tar = args.config_scale_tar
    cfg.exp_name = args.exp_name
    args = parse_arguments()

    model_id = "CompVis/stable-diffusion-v1-4"
    pipeline = StableDiffusionPipeline.from_pretrained(model_id).to(args.device)
    del pipeline.unet # Remove the old unet
    unet = UnetDownSampling.from_pretrained(model_id, subfolder="unet").to(args.device)
    pipeline.unet = unet

    # Freeze parameters
    for param in pipeline.vae.parameters():
        param.requires_grad = False
    
    for param in pipeline.unet.parameters():
        param.requires_grad = False
    
    for param in pipeline.text_encoder.parameters():
        param.requires_grad = False
    
    # Try with DDIM inversion first
    if args.mode == "ddpm_inv":
        pipeline.scheduler = DDIMScheduler.from_config(model_id, subfolder = "scheduler")
    
    pipeline.scheduler.set_timesteps(args.num_diffusion_steps)
    actor = UnetActorDiscrete(1280, 512, 2)
    critic = UnetCriticDiscrete(1280, 512)
    sft_model = copy.deepcopy(actor)
    actor.to(args.device)
    critic.to(args.device)
    sft_model.to(args.device)
    train_dataset = AutoEditDataset(args.dataset_dir, pipeline.tokenizer)

    trainer = PPOTrainerDDPMInv(cfg, actor, critic, sft_model, train_dataset, pipeline, args.device)
    trainer.fit()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)