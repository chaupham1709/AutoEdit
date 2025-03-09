from torch.utils.data import Dataset
import os
import json
from torchvision import transforms
from PIL import Image
import numpy as np
import re
import torch

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

class AutoEditDataset(Dataset):
    def __init__(self, annotation_folder, 
                 tokenizer,
                 train=True):
        super().__init__()
        self.annotation_folder = annotation_folder
        if train:
            annotation_file = os.path.join(self.annotation_folder, "all_train_val.json")
        else:
            annotation_file = os.path.join(self.annotation_folder, "val_set.json")
        
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

        src_edit_words = re.findall(r'\[(.*?)\]', src_prompt)
        tgt_edit_words = re.findall(r'\[(.*?)\]', tgt_prompt)
        src_edit_word = ""
        tgt_edit_word = ""
        for i, word in enumerate(src_edit_words):
            src_edit_word += word
            if i < len(src_edit_words) - 1:
                src_edit_word += ", "
        for i, word in enumerate(tgt_edit_words):
            tgt_edit_word += word
            if i < len(tgt_edit_words) - 1:
                tgt_edit_word += ", "

        src_prompt = re.sub(r'\[(\w+)\]', r'\1', src_prompt)
        tgt_prompt = re.sub(r'\[(\w+)\]', r'\1', tgt_prompt)
        

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
        editing_type_id = int(anns["editing_type_id"])
        if tgt_edit_word == "":
            editing_type_id = 3
        edit_prompt = anns["edit_word"]
            
        return {
            "src_input_ids": src_input_ids,
            "tgt_input_ids": tgt_input_ids,
            "images": image,
            "masks": mask,
            'editing_type_ids': editing_type_id,
            "edit_prompts": edit_prompt
        }

def collate_fn(batches):
    src_input_ids = torch.stack([batch['src_input_ids'] for batch in batches])
    tgt_input_ids = torch.stack([batch["tgt_input_ids"] for batch in batches])
    images = torch.stack([batch["images"] for batch in batches])
    editing_type_ids = torch.tensor([batch["editing_type_ids"] for batch in batches]).long()
    # src_edit_words = [batch["src_edit_words"] for batch in batches]
    # tgt_edit_words = [batch["tgt_edit_words"] for batch in batches]
    edit_prompts = [batch["edit_prompts"] for batch in batches]
    masks = torch.stack([batch["masks"] for batch in batches])
    return {
        "src_input_ids": src_input_ids,
        "tgt_input_ids": tgt_input_ids,
        "images": images,
        "masks": masks,
        "editing_type_ids": editing_type_ids,
        "edit_prompts": edit_prompts
    }