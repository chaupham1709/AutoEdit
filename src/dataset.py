from torch.utils.data import Dataset
import os
import json
from torchvision import transforms
from PIL import Image
import numpy as np
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

class AutoEditDataset(Dataset):
    def __init__(self, annotation_folder, 
                 tokenizer,
                 train=True):
        super().__init__()
        self.annotation_folder = annotation_folder
        if train:
            annotation_file = os.path.join(self.annotation_folder, "train_set.json")
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
        mask = mask_decode(mask, (image.shape[1], image.shape[2]))
        return {
            "src_input_ids": src_input_ids,
            "tgt_input_ids": tgt_input_ids,
            "image": image,
            "mask": mask
        }