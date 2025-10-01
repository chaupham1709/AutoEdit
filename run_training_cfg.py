from p2p_dataset import P2PDataset
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
import argparse
from actor_critic import UnetDownSampling, UnetActorDiscrete, UnetCriticDiscrete
import json
import numpy as np
from ddpm_p2p_with_cfg import DDPMPrompt2PromptCFG

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--logging_dir", type=str, default="log")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--checkpointing_steps", type=int, default=1)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--train_ppo", action="store_true", default=False)
    parser.add_argument("--alpha1", type=float, default=3.0)
    parser.add_argument("--alpha2", type=float, default=1.0)
    parser.add_argument("--low", type=float, default=0.05)
    parser.add_argument("--range", type=float, default=0.5)
    parser.add_argument("--metric", type=str, default="mse")
    parser.add_argument("--step_inference", type=str, default=500)
    parser.add_argument("--input_image_path", type=str)
    parser.add_argument("--src_prompt", type=str)
    parser.add_argument("--tgt_prompt", type=str)
    parser.add_argument("--save_edit_path", type=str)

    args = parser.parse_args()
    return args

def inference(args):
    ddpm_inv = DDPMPrompt2PromptCFG(args,
                                    dataset=None,
                                    device="cuda")
    ddpm_inv.init_model()
    img, _ = ddpm_inv.image_generation(
        image_path=args.input_image_path,
        src_prompt=args.src_prompt,
        tgt_prompt=args.tgt_prompt
    )
    img.save(args.save_edit_path)

args = parse_arguments()
inference(args)