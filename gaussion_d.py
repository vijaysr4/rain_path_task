import os
import math
from abc import abstractmethod
from typing import Any, List, Optional, Tuple
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import *
from model import *


class GaussianDiffusion:
    def __init__(
        self,
        timesteps: int = 1000,
        beta_schedule: str = 'linear'
        ) -> None:

        self.timesteps = timesteps

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.betas = betas

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min =1e-20))

        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def _extract(self,
                 a: torch.Tensor,
                 t: torch.Tensor,
                 x_shape: Tuple[int, ...]) -> torch.Tensor:
        # get the param of given timestep t
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    def q_sample(self,
                 x_start: torch.Tensor,
                 t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        # forward diffusion (using the nice property): q(x_t | x_0)
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise



    def q_mean_variance(self,
                        x_start: torch.Tensor,
                        t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get the mean and variance of q(x_t | x_0).
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_posterior_mean_variance(self,
                                  x_start: torch.Tensor,
                                  x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_start_from_noise(self,
                                 x_t: torch.Tensor,
                                 t: torch.Tensor,
                                 noise: torch.Tensor) -> torch.Tensor:

        # compute x_0 from x_t and pred noise: the reverse of `q_sample`

        # from x_t to x_0, Hint: Eq. 15 in DDPM

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        x0 = (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t
        return x0


    def p_mean_variance(self,
                        model: nn.Module,
                        x_t: torch.Tensor,
                        t: torch.Tensor,
                        clip_denoised: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # compute predicted mean and variance of p(x_{t-1} | x_t)
        # predict noise using model
        pred_noise = model(x_t, t)
        # get the predicted x_0: different from the algorithm2 in the paper

        # model_mean, posterior_variance, posterior_log_variance
        # the predicted mean, variance and log variance of p(x_{t-1} | x_t)
        # do not forget to clip the denoised x_0 by
        # torch.clamp(x_reconstructed, min=-1., max=1.)

        x_reconstructed = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            x_reconstructed = torch.clamp(x_reconstructed, -1., 1.)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(x_reconstructed, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self,
                 model: nn.Module,
                 x_t: torch.Tensor,
                 t: torch.Tensor,
                 clip_denoised: bool = True) -> torch.Tensor:
        # denoise_step: sample x_{t-1} from x_t and pred_noise
        # predict mean and variance
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t, clip_denoised=clip_denoised)
        noise = torch.randn_like(x_t)
        # no noise when t == 0
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        # compute x_{t-1}
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_img

    @torch.no_grad()
    def p_sample_loop(self,
                      model: nn.Module,
                      shape: Tuple[int, ...]) -> List[torch.Tensor]:
        # denoise: reverse diffusion
        batch_size = shape[0]
        device = next(model.parameters()).device
        # start from pure noise (for each example in the batch)
        # loop sample x_{t-1} from x_t
        # return the list of sampled images

        img = torch.randn(shape, device=device)
        imgs = []
        for t_ in reversed(range(self.timesteps)):
            t_tensor = torch.full((batch_size,), t_, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t_tensor, clip_denoised=True)
            # Print debug information every 50 timesteps using the integer value
            # if t_int % 50 == 0:
                # print(f"At timestep {t_int}: min {img.min().item()}, max {img.max().item()}")
                # sys.stdout.flush()
            imgs.append(img)

        return imgs

    @torch.no_grad()
    def sample(self,
               model: nn.Module,
               image_size: int,
               batch_size: int = 8,
               channels: int = 3) -> List[torch.Tensor]:
        # sample new images
        return self.p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))

    def train_losses(self,
                     model: nn.Module,
                     x_start: torch.Tensor,
                     t: torch.Tensor) -> torch.Tensor:
        """ compute train losses """
        # generate random noise
        noise = torch.randn_like(x_start)

        # 1. noised_x,
        # 2. predicted_noise,
        # 3. compare it with sampled noise

        noised_x = self.q_sample(x_start, t, noise)
        predicted_noise = model(noised_x, t)
        loss = F.mse_loss(predicted_noise, noise)
        return loss