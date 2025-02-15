import torch
from tqdm import tqdm

def encode_text(model, prompts):
    text_input = model.tokenizer(
        prompts,
        padding="max_length",
        max_length=model.tokenizer.model_max_length, 
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_encoding = model.text_encoder(text_input.input_ids.to(model.device))[0]
    return text_encoding

def sample_xts_from_x0(model, x0, num_inference_steps=50):
    """
    Samples from P(x_1:T|x_0)
    """
    # torch.manual_seed(43256465436)
    alpha_bar = model.scheduler.alphas_cumprod
    sqrt_one_minus_alpha_bar = (1-alpha_bar) ** 0.5
    alphas = model.scheduler.alphas
    batch_size = x0.shape[0]
    betas = 1 - alphas
    variance_noise_shape = (
            batch_size,
            num_inference_steps,
            model.unet.in_channels, 
            model.unet.sample_size,
            model.unet.sample_size)
    
    timesteps = model.scheduler.timesteps.to(model.device)
    t_to_idx = {int(v):k for k,v in enumerate(timesteps)}
    xts = torch.zeros((batch_size, num_inference_steps+1,model.unet.in_channels, model.unet.sample_size, model.unet.sample_size)).to(x0.device)
    xts[:,0] = x0
    for t in reversed(timesteps):
        idx = num_inference_steps-t_to_idx[int(t)]
        xts[:,idx] = x0 * (alpha_bar[t] ** 0.5) +  torch.randn_like(x0) * sqrt_one_minus_alpha_bar[t]


    return xts

def inversion_forward_process(model, x0, 
                            etas = None,    
                            prog_bar = False,
                            prompt = "",
                            cfg_scale = 3.5,
                            num_inference_steps=50, eps = None):

    if not type(prompt) is torch.Tensor:
        text_embeddings = encode_text(model, prompt)
    else:
        text_embeddings = prompt
    uncond_embedding = encode_text(model, "")
    batch_size = x0.shape[0]
    timesteps = model.scheduler.timesteps.to(model.device)
    variance_noise_shape = (
        batch_size,
        num_inference_steps,
        model.unet.in_channels, 
        model.unet.sample_size,
        model.unet.sample_size)
    if etas is None or (type(etas) in [int, float] and etas == 0):
        eta_is_zero = True
        zs = None
    else:
        eta_is_zero = False
        if type(etas) in [int, float]: etas = [etas]*model.scheduler.num_inference_steps
        xts = sample_xts_from_x0(model, x0, num_inference_steps=num_inference_steps)
        alpha_bar = model.scheduler.alphas_cumprod
        zs = torch.zeros(size=variance_noise_shape, device=model.device)
    
    t_to_idx = {int(v):k for k,v in enumerate(timesteps)}
    xt = x0
    # op = tqdm(reversed(timesteps)) if prog_bar else reversed(timesteps)
    op = tqdm(timesteps) if prog_bar else timesteps

    for t in op:
        # idx = t_to_idx[int(t)]
        idx = num_inference_steps-t_to_idx[int(t)]-1
        # 1. predict noise residual
        if not eta_is_zero:
            xt = xts[:,idx+1]
            # xt = xts_cycle[idx+1][None]
                    
        with torch.no_grad():
            _, out = model.unet.forward(xt, timestep =  t, encoder_hidden_states = uncond_embedding)
            _, cond_out = model.unet.forward(xt, timestep=t, encoder_hidden_states = text_embeddings)

        noise_pred = out.sample + cfg_scale * (cond_out.sample - out.sample)
        if eta_is_zero:
            # 2. compute more noisy image and set x_t -> x_t+1
            xt = forward_step(model, noise_pred, t, xt)

        else: 
            # xtm1 =  xts[idx+1][None]
            xtm1 =  xts[:,idx]
            # pred of x0
            pred_original_sample = (xt - (1-alpha_bar[t])  ** 0.5 * noise_pred ) / alpha_bar[t] ** 0.5
            
            # direction to xt
            prev_timestep = t - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
            alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
            
            variance = get_variance(model, t)
            pred_sample_direction = (1 - alpha_prod_t_prev - etas[idx] * variance ) ** (0.5) * noise_pred

            mu_xt = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

            z = (xtm1 - mu_xt ) / ( etas[idx] * variance ** 0.5 )
            zs[:,idx] = z

            # correction to avoid error accumulation
            xtm1 = mu_xt + ( etas[idx] * variance ** 0.5 )*z
            xts[:,idx] = xtm1
    
    if not zs is None: 
        zs[:,0] = torch.zeros_like(zs[:,0]) 

    return xt, zs, xts

def get_variance(model, timestep): #, prev_timestep):
    prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    return variance

def forward_step(model, model_output, timestep, sample):
    next_timestep = min(model.scheduler.config.num_train_timesteps - 2,
                        timestep + model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps)

    # 2. compute alphas, betas
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    # alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep] if next_ltimestep >= 0 else self.scheduler.final_alpha_cumprod

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

    # 5. TODO: simple noising implementatiom
    next_sample = model.scheduler.add_noise(pred_original_sample,
                                    model_output,
                                    torch.LongTensor([next_timestep]))
    return next_sample

def reverse_step(model, model_output, timestep, sample, eta = 0, variance_noise=None):
    # 1. get previous step value (=t-1)
    prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    # 2. compute alphas, betas
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)    
    # variance = self.scheduler._get_variance(timestep, prev_timestep)
    variance = get_variance(model, timestep) #, prev_timestep)
    std_dev_t = eta * variance ** (0.5)
    # Take care of asymetric reverse process (asyrp)
    model_output_direction = model_output
    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    # pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output_direction
    pred_sample_direction = (1 - alpha_prod_t_prev - eta * variance) ** (0.5) * model_output_direction
    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
    # 8. Add noice if eta > 0
    if eta > 0:
        if variance_noise is None:
            variance_noise = torch.randn(model_output.shape, device=model.device)
        sigma_z =  eta * variance ** (0.5) * variance_noise
        prev_sample = prev_sample + sigma_z

    return prev_sample