import torch
from transformers import AutoProcessor, AutoTokenizer
from .mask_clip_model import MaskCLIPModel
from torch.nn import functional as F
from torchmetrics.multimodal import CLIPScore
import numpy as np
from transformers import CLIPModel, CLIPProcessor

def change_mask_to_attention_mask(mask, patch_size=16, image_size=224, use_cls_token=True):
    h, w = image_size // patch_size, image_size // patch_size
    seq_len = h * w + 1 if use_cls_token else h * w
    # mask.shape -> (H, W)
    mask = mask.unsqueeze(0).unsqueeze(0)
    # mask.shape -> (N=1, C=1, H, W)
    mask = torch.nn.functional.interpolate(
            mask.float(), 
            size=(h, w),
            mode="nearest",
        ) > 0.5
    # mask.shape -> (N=1, H, W)
    mask = mask.squeeze(1)
    mask = torch.flatten(mask, start_dim=1)
    cls_token = torch.ones_like(mask[:, :1])
    mask = torch.concat((cls_token, mask), dim=1)
    attention_mask = torch.zeros((1, 1, seq_len, seq_len))
    for i in range(seq_len):
        for j in range(seq_len):
            if (mask[:, i] and mask[:, j]):
                attention_mask[:, :, i, j] = 1
    return attention_mask

def change_mask_shape(mask, image):
    _, _, h, w = image.shape
    mask = mask.unsqueeze(1).repeat(1, 3, 1, 1)
    new_mask = torch.nn.functional.interpolate(
        mask.float(), 
        size=(h, w),
        mode="nearest",
    ) > 0.5
    return new_mask

def non_editing_region_negative_MSE(
    origin_image,
    editing_region_mask,
    candidate_images,
    clip_text_path,
):
    scores = []
    processor = AutoProcessor.from_pretrained(clip_text_path)
    origin_image = processor(images=origin_image, return_tensors="pt")['pixel_values']
    non_editing_region_mask = change_mask_shape(1 - editing_region_mask, origin_image)
    non_editing_region_of_origin_image = origin_image * non_editing_region_mask
    for i in range(len(candidate_images)):
        target_image = processor(images=candidate_images[i], return_tensors='pt')['pixel_values']
        non_editing_region_of_target_image = target_image * non_editing_region_mask
        score = -F.mse_loss(non_editing_region_of_origin_image, non_editing_region_of_target_image)
        scores.append(score.item())
    return scores

def editing_region_clip_score(
    target_prompt_change,
    candidate_images,
    editing_region_mask,
    clip_text_path,
):
    scores = []
    model = MaskCLIPModel.from_pretrained(clip_text_path)
    processor = AutoProcessor.from_pretrained(clip_text_path)
    tokenizer = AutoTokenizer.from_pretrained(clip_text_path)
    target_prompt_change = tokenizer(target_prompt_change, padding=True, return_tensors="pt")
    target_prompt_change_feature = model.get_text_features(**target_prompt_change)
    if clip_text_path.split('/')[-1] == "clip-vit-base-patch16":
        editing_region_self_attention_mask = change_mask_to_attention_mask(editing_region_mask)
    elif clip_text_path.split('/')[-1] == "clip-vit-large-patch14":
        editing_region_self_attention_mask = change_mask_to_attention_mask(editing_region_mask, patch_size=14)
        
    for i in range(len(candidate_images)):
        target_image = processor(images=candidate_images[i], return_tensors="pt")
        editing_region_of_target_image_feature = model.get_mask_image_features(**target_image, image_attention_mask=editing_region_self_attention_mask)
        score = torch.cosine_similarity(target_prompt_change_feature, editing_region_of_target_image_feature)
        scores.append(score.item())
    return scores

class RewardFunction:
    def __init__(self, lambd=0.4):
        self.lambd = lambd
        self.clip_text_path = "openai/clip-vit-base-patch16"
        self.clip_mask_model = MaskCLIPModel.from_pretrained(self.clip_text_path)
        self.clip_processor = AutoProcessor.from_pretrained(self.clip_text_path)
        self.clip_tokenizer = AutoTokenizer.from_pretrained(self.clip_text_path)
        self.clip_model = CLIPModel.from_pretrained(self.clip_text_path)
        self.processor = CLIPProcessor.from_pretrained(self.clip_text_path)
    
    @torch.no_grad()
    def editing_region_clip_score(self,
                                  target_prompt_change,
                                  candidate_images,
                                  editing_region_mask
                                  ):
        target_prompt_change = self.clip_tokenizer(target_prompt_change, padding=True, return_tensors="pt")
        target_prompt_change_feature = self.clip_mask_model.get_text_features(**target_prompt_change)
        editing_region_self_attention_masks = []
        for region_mask in editing_region_mask:
            if self.clip_text_path.split('/')[-1] == "clip-vit-base-patch16":
                editing_region_self_attention_mask = change_mask_to_attention_mask(region_mask)
            elif self.clip_text_path.split('/')[-1] == "clip-vit-large-patch14":
                editing_region_self_attention_mask = change_mask_to_attention_mask(region_mask, patch_size=14)

            editing_region_self_attention_masks.append(editing_region_self_attention_mask)
        
        scores = []
        for i in range(len(candidate_images)):
            target_image = self.clip_processor(images=candidate_images[i], return_tensors="pt")
            editing_region_of_target_image_feature = self.clip_mask_model.get_mask_image_features(**target_image, image_attention_mask=editing_region_self_attention_masks[i])
            score = torch.cosine_similarity(target_prompt_change_feature[i], editing_region_of_target_image_feature)
            scores.append(score)
        return torch.cat(scores)
        # if type(editing_region_mask) == torch.Tensor: editing_region_mask = editing_region_mask.cpu().numpy()
        # scores = []
        # for img, prompt_change, region_mask in zip(candidate_images, target_prompt_change, editing_region_mask):
        #     img = img.astype(np.float32)
        #     img = region_mask[:,:,None] * img
        #     img = img.astype(np.uint8)
        #     inputs = self.processor(text=[prompt_change], images=img, return_tensors="pt", padding=True)
        #     outputs = self.clip_model(**inputs)
        #     logits_per_image = outputs.logits_per_image / self.clip_model.logit_scale.exp()
        #     scores.append(logits_per_image.item())
        # return torch.tensor(scores)

    @torch.no_grad()
    def non_editing_region_negative_MSE(self,
                                        origin_images,
                                        editing_region_masks,
                                        candidate_images, 
                                        ):
        scores = []
        origin_images = self.clip_processor(images=origin_images, return_tensors="pt")['pixel_values']
        non_editing_region_mask = change_mask_shape(1 - editing_region_masks, origin_images)
        non_editing_region_of_origin_image = origin_images * non_editing_region_mask
        target_image = self.clip_processor(images=candidate_images, return_tensors="pt")["pixel_values"]
        non_editing_region_of_target_image = target_image * non_editing_region_mask
        scores = -F.mse_loss(non_editing_region_of_origin_image, non_editing_region_of_target_image, reduction='none')
        scores = torch.mean(scores, dim=[1,2,3])
        return scores

    
    def estimate_reward(
            self,
            origin_images,
            editing_region_masks,
            editing_images,
            editing_types,
            prompt_changes,
    ):
        non_region_score = self.non_editing_region_negative_MSE(
            origin_images,
            editing_region_masks,
            editing_images
        )
        editing_region_score = self.editing_region_clip_score(
            prompt_changes,
            editing_images,
            editing_region_masks
        )
        delete_editing_mask = (editing_types == 3).float()

        non_deleting_score = editing_region_score + self.lambd * non_region_score
        deleting_score = (0.5 - editing_region_score) + self.lambd * non_region_score
        total_score = delete_editing_mask * deleting_score + (1-delete_editing_mask) * non_deleting_score
        
        return total_score