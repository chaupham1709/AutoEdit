
from torch.utils.data import Dataset
import os
import json
from PIL import Image
from torchvision import transforms
import random
import numpy as np
import torch

class AutoEditDataset(Dataset):
    def __init__(self, image_path, annotation_file, tokenizer):
        super().__init__()
        self.image_path = image_path

        with open(os.path.join(self.image_path, annotation_file), "r") as f:
            self.annotations = json.load(f)
        
        self.image_size = 512
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.annotations)
    
    def cropped_image(self, img, bbox=[]):
        img_height = img.height; img_width = img.width
        if img_height > img_width:
            cropped_size = img_width
            c_l = 0; c_r = img_width
            if len(bbox) > 0:
                left, top, right, bottom = bbox
                top_borderline = int(max(0, bottom - cropped_size))
                bottom_borderline = int(min(top, img_height - cropped_size))
                if bottom_borderline > top_borderline:
                    c_t = random.randint(top_borderline, bottom_borderline)
                    c_b = c_t + cropped_size
                else:
                    c_t = top_borderline
                    c_b = c_t + cropped_size
            else:
                c_t = random.randint(0, img_height - cropped_size)
                c_b = c_t + cropped_size

        else:
            cropped_size = img_height
            c_t = 0; c_b = img_height
            if len(bbox) > 0:
                left, top, right, bottom = bbox
                left_borderline = int(max(0, right - cropped_size))
                right_borderline = int(min(left, img_width - cropped_size))
                if right_borderline > left_borderline:
                    c_l = random.randint(left_borderline, right_borderline)
                    c_r = c_l + cropped_size
                else:
                    c_l = left_borderline
                    c_r = c_l + cropped_size
            else:
                c_l = random.randint(0, img_width - cropped_size)
                c_r = c_l + cropped_size
        
        return c_l, c_t, c_r, c_b


    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_file = os.path.join(self.image_path, ann['image'])
        img = Image.open(image_file).convert("RGB")
        img_width, img_height = img.width, img.height

        if "bbox" in ann:
            bbox = ann["bbox"]
        else:
            bbox = []
        c_l, c_t, c_r, c_b = self.cropped_image(img, bbox)
        img_cropped = img.crop((c_l, c_t, c_r, c_b)).resize((self.image_size, self.image_size))
        mask_file = os.path.join(self.image_path, ann['mask'])
        mask = Image.open(mask_file).convert("L")
        mask_cropped = mask.crop((c_l, c_t, c_r, c_b)).resize((self.image_size, self.image_size))

        img_tensor = self.image_transform(img_cropped)
        mask_tensor = torch.from_numpy(np.array(mask_cropped))
        mask_tensor = (mask_tensor > 5).float()

        src_prompt = ann["src_prompt"]
        tgt_prompt = ann["edit_prompt"]
        input_ids = self.tokenizer(
            [src_prompt, tgt_prompt],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        ).input_ids

        src_input_ids = input_ids[0]; tgt_input_ids = input_ids[1]

        return {
            "images": img_tensor,
            "masks": mask_tensor,
            "src_input_ids": src_input_ids,
            "tgt_input_ids": tgt_input_ids,
            "edit_prompts": tgt_prompt
        }

def collate_fn(batch):
    images = torch.stack([item['images'] for item in batch], dim=0)
    masks = torch.stack([item['masks'] for item in batch], dim=0)
    src_input_ids = torch.stack([item['src_input_ids'] for item in batch], dim=0)
    tgt_input_ids = torch.stack([item['tgt_input_ids'] for item in batch], dim=0)
    edit_prompts = [item['edit_prompts'] for item in batch]

    return {
        "images": images,
        "masks": masks,
        "src_input_ids": src_input_ids,
        "tgt_input_ids": tgt_input_ids,
        "edit_prompts": edit_prompts
    }
