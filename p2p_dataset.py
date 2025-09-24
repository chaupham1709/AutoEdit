from torch.utils.data import Dataset
import os
import json
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
from prompt_to_prompt.seq_aligner import get_refinement_mapper
from prompt_to_prompt.ptp_utils import get_word_inds
import re

def mask_decode(encoded_mask,image_shape=[512,512]):
    length=image_shape[0]*image_shape[1]
    mask_array=np.zeros((length,))
    
    for i in range(0,len(encoded_mask),2):
        splice_len=min(encoded_mask[i+1],length-encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i]+j]=1
            
    mask_array=mask_array.reshape(image_shape[0], image_shape[1])
    # to avoid annotation errors in boundary
    mask_array[0,:]=1
    mask_array[-1,:]=1
    mask_array[:,0]=1
    mask_array[:,-1]=1
            
    return mask_array

def get_equalizer(text, word_select, tokenizer):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.zeros((77,))
    # values = torch.tensor(values, dtype=torch.float32)
    for word in word_select:
        inds = get_word_inds(text, word, tokenizer)
        equalizer[inds] = 1
    return equalizer

class P2PDataset(Dataset):
    def __init__(self, annotation_folder,
                 tokenizer):
        super().__init__()
        self.annotation_folder = annotation_folder
        annotation_file = os.path.join(self.annotation_folder, "all_train_val.json")

        with open(annotation_file, "r") as f:
            self.annotation_data = json.load(f)
        
        self.tokenizer = tokenizer
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.annotation_data)
    
    def __getitem__(self, idx):
        anns = self.annotation_data[idx]
        img_file = os.path.join(self.annotation_folder, "annotation_images", anns["image_path"])
        src_prompt = anns["original_prompt"]
        tgt_prompt = anns["editing_prompt"]

        matches = re.findall(r"\[(.*?)\]", tgt_prompt)

        src_prompt = src_prompt.replace("[", "")
        src_prompt = src_prompt.replace("]", "")
        tgt_prompt = tgt_prompt.replace("[", "")
        tgt_prompt = tgt_prompt.replace("]", "")
        equalizer = get_equalizer(tgt_prompt, matches, self.tokenizer)

        src_tar_len_eq = (len(src_prompt.split(" ")) == len(tgt_prompt.split(" ")))
        if src_tar_len_eq:
            replacement = True
            mapper = torch.arange(0, 77).long().unsqueeze(0)
            alphas = torch.ones((1, 77)).float()
        else:
            mapper, alphas = get_refinement_mapper([src_prompt, tgt_prompt], self.tokenizer)
            replacement = False
        
        input_ids = self.tokenizer(
            [src_prompt, tgt_prompt],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        ).input_ids
        
        src_input_ids = input_ids[0]; tgt_input_ids = input_ids[1]
        image = Image.open(img_file).convert("RGB")
        image = self.image_transform(image)
        mask = anns["mask"]
        mask = torch.from_numpy(mask_decode(mask, (image.shape[1], image.shape[2]))).float()
        edit_prompt = tgt_prompt

        return {
            "src_input_ids": src_input_ids,
            "tgt_input_ids": tgt_input_ids,
            "images": image,
            "masks": mask,
            "edit_prompts": edit_prompt,
            "mapper": mapper.squeeze(0),
            "alphas": alphas.squeeze(0),
            "replacement": replacement,
            "equalizer": equalizer
        }

def collate_fn(batches):
    src_input_ids = torch.stack([batch["src_input_ids"] for batch in batches])
    tgt_input_ids = torch.stack([batch["tgt_input_ids"] for batch in batches])
    images = torch.stack([batch["images"] for batch in batches])
    edit_prompts = [batch["edit_prompts"] for batch in batches]
    masks = torch.stack([batch["masks"] for batch in batches])
    mapper = torch.stack([batch["mapper"] for batch in batches])
    alphas = torch.stack([batch["alphas"] for batch in batches])
    replacement = torch.tensor([batch["replacement"] for batch in batches])
    equalizer = torch.stack([batch["equalizer"] for batch in batches])
    return {
        "src_input_ids": src_input_ids,
        "tgt_input_ids": tgt_input_ids,
        "images": images,
        "edit_prompts": edit_prompts,
        "masks": masks,
        "mapper": mapper,
        "alphas": alphas,
        "replacement": replacement,
        "equalizer": equalizer
    }
