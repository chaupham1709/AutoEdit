from autoedit_dataset import AutoEditDataset
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
import argparse
from ddpm_p2p_wo_attention import Prompt2PromptDDPM, Prompt2PromptDDPMPPO
from actor_critic import UnetDownSampling, UnetActorDiscrete, UnetCriticDiscrete

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
    parser.add_argument("--train_sft", action="store_true", default=False)
    parser.add_argument("--alpha1", type=float, default=3.0)
    parser.add_argument("--alpha2", type=float, default=3.0)
    parser.add_argument("--low", type=float, default=0.05)
    parser.add_argument("--range", type=float, default=0.5)
    parser.add_argument("--metric", type=str, default="mse")
    parser.add_argument("--step_inference", type=str, default=500)
    parser.add_argument("--input_image_path", type=str)
    parser.add_argument("--src_prompt", type=str)
    parser.add_argument("--tgt_prompt", type=str)
    parser.add_argument("--save_edit_path", type=str)
    parser.add_argument("--annotation_folder", type=str)

    args = parser.parse_args()
    return args

def train_first_stage(args):
    model_id = "CompVis/stable-diffusion-v1-4"
    pipeline = StableDiffusionPipeline.from_pretrained(model_id).to("cuda")
    pipeline.unet = UnetDownSampling.from_pretrained(model_id, subfolder="unet").to("cuda")
    pipeline.scheduler = DDIMScheduler.from_config(model_id, subfolder="scheduler")
    actor = UnetActorDiscrete(1280*2, 768, 2, 2)
    dataset = AutoEditDataset(
        args.annotation_folder,
        annotation_file="editbench_full.json",
        tokenizer=pipeline.tokenizer
    )
    p2p_ddpm = Prompt2PromptDDPM(
        args,
        dataset=dataset,
        device="cuda",
        alpha1=args.alpha1,
        alpha2=args.alpha2
    )
    p2p_ddpm.train_first_stage()

def train_ppo(args):
    from tqdm import tqdm
    torch.manual_seed(42)
    args = parse_arguments()
    model_id = "CompVis/stable-diffusion-v1-4"
    pipeline = StableDiffusionPipeline.from_pretrained(model_id).to("cuda")
    tokenizer = pipeline.tokenizer
    del pipeline.unet
    del pipeline.text_encoder
    del pipeline.vae

    dataset = AutoEditDataset(
        args.annotation_folder,
        annotation_file="editbench_full.json",
        tokenizer=tokenizer
    )

    p2p_ppo = Prompt2PromptDDPMPPO(
        args,
        dataset,
        device="cuda",
        alpha1=3.0,
        alpha2=3.0,
    )
    p2p_ppo.fit_ppo()

@torch.no_grad()
def model_inference(args):
    ddpm_inv = Prompt2PromptDDPM(
        args,
        dataset=None,
        device="cuda",
        alpha1=3.0,
        alpha2=3.0
    )
    ddpm_inv.init_model()
    img, _ = ddpm_inv.image_generation(
        image_path=args.input_image_path,
        src_prompt=args.src_prompt,
        tgt_prompt=args.tgt_prompt
    )
    img.save(args.save_edit_path)

if __name__ == "__main__":
    args = parse_arguments()
    if args.train_ppo:
        train_ppo(args)
    elif args.train_sft:
        train_first_stage(args)
    else:
        model_inference(args)