import torch.nn as nn
from diffusers import UNet2DConditionModel
import torch
from typing import Union, Optional, Dict, Any, Tuple
from diffusers.models.unet_2d_condition import UNet2DConditionModel, UNet2DConditionOutput
from torch.nn import functional as F
import math
from typing import Optional

class UnetDownSampling(UNet2DConditionModel):
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        return_hidden=False
    ) -> Union[UNet2DConditionOutput, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples
        
        if return_hidden:
            return_features = sample

        # 4. mid
        sample = self.mid_block(
            sample,
            emb,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
        )

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )
        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)
        
        if return_hidden:
            return return_features, UNet2DConditionOutput(sample=sample)

        return UNet2DConditionOutput(sample=sample)


def get_timestep_encoding(t, dim):
    """
    Sinusoidal timestep encoding used in DDPM-style diffusion models.
    
    Args:
        t (torch.Tensor): Tensor of shape (batch_size,) or scalar with timesteps.
        dim (int): Dimension of the encoding.
    
    Returns:
        torch.Tensor: Encoded timestep tensor of shape (batch_size, dim)
    """
    if len(t.shape) == 0:
        t = t[None]  # Convert scalar to 1D tensor

    half_dim = dim // 2
    exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32) / half_dim
    freqs = torch.exp(exponent).to(t.device)
    angles = t[:, None].float() * freqs[None]
    encoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    
    if dim % 2 == 1:
        encoding = torch.nn.functional.pad(encoding, (0, 1))  # pad to odd dim if needed

    return encoding

class UnetActorDiscrete(nn.Module):
    def __init__(self, in_dim, hidden_dim, output_dim1, output_dim2):
        super().__init__()
        self.mlp1 = nn.Linear(in_dim, hidden_dim // 2)
        self.mlp2 = nn.Linear(hidden_dim, hidden_dim)
        self.pos_enc_dim = 128
        
        self.time_mlp = nn.Linear(self.pos_enc_dim, hidden_dim // 2)
        self.output_layer1 = nn.Linear(hidden_dim, output_dim1)
        self.output_layer2 = nn.Linear(hidden_dim, output_dim2)
    
    def forward(self, x, t):
        assert len(x.shape) == 4
        x = x.mean(dim=[2,3]) # Average pooling
        time_encoding = get_timestep_encoding(t, self.pos_enc_dim)
        img_features = self.mlp1(x)
        time_features = self.time_mlp(time_encoding.to(x.device))
        features = torch.cat([img_features, time_features], dim=-1)
        features = torch.relu(self.mlp2(features))

        logits1 = self.output_layer1(features)
        logits2 = self.output_layer2(features)
        
        return logits1, logits2
    
    def forward_actor(self, x, t, gt_action1, gt_action2):
        batch_size, num_timesteps = x.shape[:2]
        x = x.reshape(batch_size * num_timesteps, x.shape[2], x.shape[3], -1)
        logits1, logits2 = self.forward(x, t)
        logits1 = logits1.reshape(batch_size, num_timesteps, -1)
        log_prob_all_action1 = F.log_softmax(logits1, dim=-1)

        gt_action1 = gt_action1.unsqueeze(-1)
        log_prob_output1 = log_prob_all_action1.gather(dim=2, index=gt_action1)

        if gt_action2 is not None:

            logits2 = logits2.reshape(batch_size, num_timesteps, -1)
            log_prob_all_action2 = F.log_softmax(logits2, dim=-1)

            gt_action2 = gt_action2.unsqueeze(-1)
            log_prob_output2 = log_prob_all_action2.gather(dim=2, index=gt_action2)
            return log_prob_output1, log_prob_output2
        
        return log_prob_output1, None

class UnetActorWeighting(nn.Module):
    def __init__(self, in_dim, hidden_dim, output_dim1, output_dim2, num_weighting):
        super().__init__()
        self.mlp1 = nn.Linear(in_dim, hidden_dim // 2)
        self.mlp2 = nn.Linear(hidden_dim, hidden_dim)
        self.pos_enc_dim = 128
        
        self.time_mlp = nn.Linear(self.pos_enc_dim, hidden_dim // 2)
        self.output_layer1 = nn.Linear(hidden_dim, output_dim1)
        self.output_layer2 = nn.Linear(hidden_dim, output_dim2)
        self.weight_layer = nn.Linear(hidden_dim, num_weighting)
    
    def forward(self, x, t):
        assert len(x.shape) == 4
        x = x.mean(dim=[2,3]) # Average pooling
        time_encoding = get_timestep_encoding(t, self.pos_enc_dim)
        img_features = self.mlp1(x)
        time_features = self.time_mlp(time_encoding.to(x.device))
        features = torch.cat([img_features, time_features], dim=-1)
        features = torch.relu(self.mlp2(features))

        logits1 = self.output_layer1(features)
        logits2 = self.output_layer2(features)
        weight_logits = self.weight_layer(features)
        
        return logits1, logits2, weight_logits
    
    def forward_actor(self, x, t, gt_action1, gt_action2):
        batch_size, num_timesteps = x.shape[:2]
        x = x.reshape(batch_size * num_timesteps, x.shape[2], x.shape[3], -1)
        logits1, logits2, weight_logits = self.forward(x, t)
        logits1 = logits1.reshape(batch_size, num_timesteps, -1)
        log_prob_all_action1 = F.log_softmax(logits1, dim=-1)

        gt_action1 = gt_action1.unsqueeze(-1)
        log_prob_output1 = log_prob_all_action1.gather(dim=2, index=gt_action1)

        logits2 = logits2.reshape(batch_size, num_timesteps, -1)
        log_prob_all_action2 = F.log_softmax(logits2, dim=-1)

        gt_action2 = gt_action2.unsqueeze(-1)
        log_prob_output2 = log_prob_all_action2.gather(dim=2, index=gt_action2)

        weight_logits = weight_logits.reshape(batch_size, num_timesteps, -1)
        log_prob_all_weight = F.log_softmax(weight_logits, dim=-1)
        weight_index = torch.ones(log_prob_output2.shape[:2]).to(logits2.device).long()
        log_prob_weight = log_prob_all_weight.gather(dim=2, index=weight_index.unsqueeze(-1))
        return log_prob_output1, log_prob_output2, log_prob_weight

class UnetCriticDiscrete(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.mlp1 = nn.Linear(in_dim, hidden_dim // 2)
        self.mlp2 = nn.Linear(hidden_dim, hidden_dim)
        self.mlp3 = nn.Linear(hidden_dim, 1)
        self.pos_enc_dim = 128
        
        self.time_mlp = nn.Linear(self.pos_enc_dim, hidden_dim // 2)
    
    def forward(self, x, t):
        assert len(x.shape) == 4
        x = x.mean(dim=[2,3]) # Average pooling
        time_encoding = get_timestep_encoding(t, self.pos_enc_dim)
        img_features = self.mlp1(x)
        time_features = self.time_mlp(time_encoding.to(x.device))
        features = torch.cat([img_features, time_features], dim=-1)
        return self.mlp3(torch.relu(self.mlp2(features)))
    
    def forward_critic(self, x, t):
        batch_size, num_timesteps = x.shape[:2]
        x = x.reshape(batch_size * num_timesteps, x.shape[2], x.shape[3], -1)
        critic_output = self.forward(x, t).squeeze(-1)
        critic_output = critic_output.reshape(batch_size, num_timesteps, -1)
        return critic_output.mean(dim=1)