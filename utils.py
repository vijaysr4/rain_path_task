import torch

def linear_beta_schedule(timesteps, start_scale=0.0001, end_scale=0.02):
    """
    beta schedule
    """

    return torch.linspace(start_scale, end_scale, timesteps)



def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    """
    steps = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos(((steps / timesteps) + s) / (1 + s) * (torch.pi / 2)) ** 2
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    return betas.float()