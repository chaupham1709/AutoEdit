from p2p_dataset import P2PDataset, collate_fn
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
import argparse
from ddpm_p2p import Prompt2PromptDDPM, Prompt2PromptDDPMPPO
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
    parser.add_argument("--annotation_folder", type=str)
    parser.add_argument("--input_image_path", type=str)
    parser.add_argument("--src_prompt", type=str)
    parser.add_argument("--tgt_prompt", type=str)
    parser.add_argument("--save_edit_path", type=str)
    parser.add_argument("--train_sft", action="store_true", default=False)

    args = parser.parse_args()
    return args

def train_first_stage(args):
    torch.manual_seed(42)
    model_id = "CompVis/stable-diffusion-v1-4"
    pipeline = StableDiffusionPipeline.from_pretrained(model_id).to("cuda")
    del pipeline.unet
    pipeline.unet = UnetDownSampling.from_pretrained(model_id, subfolder="unet").to("cuda")
    pipeline.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    actor = UnetActorDiscrete(1280*2, 768, 2, 2)

    dataset = P2PDataset(
        annotation_folder=args.annotation_folder,
        tokenizer=pipeline.tokenizer
    )
    p2p_ddpm = Prompt2PromptDDPM(
        args,
        pipeline=pipeline,
        dataset=dataset,
        device="cuda",
        actor=actor
    )
    p2p_ddpm.train_first_stage()

def train_ppo(args):
    print("PPO training")
    torch.manual_seed(42)
    args = parse_arguments()

    model_id = "CompVis/stable-diffusion-v1-4"
    pipeline = StableDiffusionPipeline.from_pretrained(model_id).to("cuda")
    tokenizer = pipeline.tokenizer
    del pipeline.unet
    del pipeline.text_encoder
    del pipeline.vae

    dataset = P2PDataset(
        annotation_folder=args.annotation_folder,
        tokenizer=tokenizer
    )
    
    p2p_ppo = Prompt2PromptDDPMPPO(
        args,
        dataset,
        device="cuda"
    )
    p2p_ppo.fit_ppo()

@torch.no_grad()
def model_inference(args):
    p2p = Prompt2PromptDDPM(
        args,
        dataset=None,
        device="cuda"
    )
    p2p.init_model()
    img, _ = p2p.image_generation(
        image_path=args.input_image_path,
        src_prompt=args.src_prompt,
        tgt_prompt=args.tgt_prompt
    )
    img.save(args.save_edit_path)

if __name__ == "__main__":
    args = parse_arguments()
    if args.train_sft:
        train_first_stage(args)
    elif args.train_ppo:
        train_ppo(args)
    else:
        model_inference(args)