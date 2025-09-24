from torchmetrics.multimodal import CLIPScore
from torchmetrics.regression import MeanSquaredError
import numpy as np
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

class RewardFunction:
    def __init__(self, alpha1, alpha2, device):
        self.device = device
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.clip_metric_calculator = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)
    
    @torch.no_grad()
    def calculate_clip_similarity(self, imgs, txts, masks=None):
        """
        Expect:
        img: np.array
        txts: list
        masks: np.array
        """
        clip_scores = []
        assert type(txts) is list
        for i, (img, txt) in enumerate(zip(imgs, txts)):
            if masks is not None:
                mask = masks[i]
                img = np.uint8(img * mask)
            img_tensor=torch.tensor(img).permute(2,0,1).to(self.device)
            score = self.clip_metric_calculator(img_tensor, txt)
            clip_scores.append(score.item())
        
        return torch.tensor(clip_scores)
    
    @torch.no_grad()
    def calculate_mse(self, img_preds, img_gts, mask_preds=None, mask_gts=None):
        img_preds = img_preds.astype(np.float32)/255
        img_gts = img_gts.astype(np.float32)/255
        assert img_preds.shape == img_gts.shape, "Image shapes should be the same."
        if mask_preds.sum() == 0: return 0.

        if mask_preds is not None:
            mask_preds = np.array(mask_preds).astype(np.float32)
            img_preds = img_preds * mask_preds
        if mask_gts is not None:
            mask_gts = np.array(mask_gts).astype(np.float32)
            img_gts = img_gts * mask_gts
        
        mse_scores = []
        for img_pred, img_gt in zip(img_preds, img_gts):
            img_pred_tensor=torch.tensor(img_pred).permute(2,0,1).to(self.device)
            img_gt_tensor=torch.tensor(img_gt).permute(2,0,1).to(self.device)
            
            score =  ((img_pred_tensor - img_gt_tensor)**2).sum() / mask_preds.sum()
            mse_scores.append(score.item())
        
        return torch.tensor(mse_scores)
    
    @torch.no_grad()
    def estimate_reward(self, img_gt, img_pred, edit_prompt, edit_mask):
        if (np.sum(1-edit_mask) == 0): negative_mse = 0
        else:
            negative_mse = -self.calculate_mse(img_pred, img_gt, 1-edit_mask, 1-edit_mask)
        clip_score = self.calculate_clip_similarity(img_pred, edit_prompt, edit_mask)
        # print(negative_mse, clip_score)
        # print(f"Negative mse: {negative_mse}. Clip score {clip_score}")
        return self.alpha1*clip_score/10 + self.alpha2*negative_mse*10

class RewardWithPSNR(RewardFunction):
    def __init__(self, alpha1, alpha2, device):
        self.device = device
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.clip_metric_calculator = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)
        self.psnr_metric_calculator = PeakSignalNoiseRatio(data_range=1.0).to(device)
    
    @torch.no_grad()
    def calculate_psnr(self, img_pred, img_gt, mask):
        """
        img_pred and img_gt are np.float32
        """
        img_pred = img_pred.astype(np.float32) / 255
        img_gt = img_gt.astype(np.float32) / 255

        if mask is not None:
            mask = np.array(mask).astype(np.float32)
            img_pred = img_pred * mask
            img_gt = img_gt * mask
        
        psnr_scores = []
        for i in range(len(img_pred)):
            img_pred_tensor = torch.tensor(img_pred[i]).permute(2,0,1).unsqueeze(0).to(self.device)
            img_gt_tensor = torch.tensor(img_gt[i]).permute(2,0,1).unsqueeze(0).to(self.device)
            psnr_scores.append(self.psnr_metric_calculator(img_pred_tensor, img_gt_tensor).cpu().item())
            if psnr_scores[-1] == float("inf"): psnr_scores[-1] = 0
        
        return torch.tensor(psnr_scores)
    
    @torch.no_grad()
    def estimate_reward(self, img_gt, img_pred, edit_prompt, edit_mask):
        if (np.sum(1-edit_mask) == 0): psnr_score = 0
        else:
            psnr_score = self.calculate_psnr(img_pred, img_gt, 1-edit_mask)
        
        clip_score = self.calculate_clip_similarity(img_pred, edit_prompt, edit_mask)
        print(clip_score, psnr_score)
        # clip_score = torch.clamp(clip_score, max=29.)
        # psnr_score = torch.clamp(psnr_score, max=28.)
        return self.alpha1 * clip_score / 10 + self.alpha2 * psnr_score / 10

class RewardWithSSIM(RewardFunction):
    def __init__(self, alpha1, alpha2, device):
        self.device = device
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.clip_metric_calculator = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)
        self.ssim_metric_calculator = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    @torch.no_grad()
    def calculate_ssim(self, img_preds, img_gts, mask_preds=None, mask_gts=None):
        img_preds = np.array(img_preds).astype(np.float32)/255
        img_gts = np.array(img_gts).astype(np.float32)/255
        assert img_preds.shape == img_gts.shape, "Image shapes should be the same."

        if mask_preds is not None:
            mask_preds = np.array(mask_preds).astype(np.float32)
            img_preds = img_preds * mask_preds
        if mask_gts is not None:
            mask_gts = np.array(mask_gts).astype(np.float32)
            img_gts = img_gts * mask_gts
        
        ssim_scores = []
        
        for img_pred, img_gt in zip(img_preds, img_gts):
            
            img_pred_tensor=torch.tensor(img_pred).permute(2,0,1).unsqueeze(0).to(self.device)
            img_gt_tensor=torch.tensor(img_gt).permute(2,0,1).unsqueeze(0).to(self.device)

            
            score =  self.ssim_metric_calculator(img_pred_tensor,img_gt_tensor)
            ssim_scores.append(score.cpu().item())
        
        return torch.tensor(ssim_scores)
    
    @torch.no_grad()
    def estimate_reward(self, img_gt, img_pred, edit_prompt, edit_mask):
        if (np.sum(1-edit_mask) == 0): ssim_score = 0
        else:
            ssim_score = self.calculate_ssim(img_pred, img_gt, 1-edit_mask, 1-edit_mask)
        
        clip_score = self.calculate_clip_similarity(img_pred, edit_prompt, edit_mask)
        return self.alpha1 * clip_score / 10 + self.alpha2 * ssim_score * 10