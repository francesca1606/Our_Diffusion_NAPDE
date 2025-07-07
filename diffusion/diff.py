#!/usr/bin/env python3
from copy import deepcopy
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from tqdm import trange
import torch.nn as nn
from diffusion.metrics import MSE,snr,gof, log_spectral_distance
from diffusion.diffusion_model import DiffusionAttnUnet1D
from diffusion.utils import ema_update
from einops import rearrange
import matplotlib.pyplot as plt
from diffusion.correlation import Loss_Covariant




# Define the noise schedule and sampling loop
def get_alphas_sigmas(t):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

def get_crash_schedule(t):
    sigma = torch.sin(t * math.pi / 2) ** 2
    alpha = (1 - sigma ** 2) ** 0.5
    return alpha_sigma_to_t(alpha, sigma)

def alpha_sigma_to_t(alpha, sigma):
    """Returns a timestep, given the scaling factors for the clean image and for
    the noise."""
    return torch.atan2(sigma, alpha) / math.pi * 2

@torch.no_grad()
def sample(model, noise, cond = None,steps = 1000, eta=0, training=False, device = torch.device("cpu")):
    """Draws samples from a model given starting noise."""
    x = noise
    ts = x.new_ones([x.shape[0]]).to(device)
    # Create the noise schedule
    t = torch.linspace(1, 0, steps + 1)[:-1].to(device)

    t = get_crash_schedule(t)

    alphas, sigmas = get_alphas_sigmas(t)
    alphas = alphas.to(device)
    sigmas = sigmas.to(device)
    generator = range(steps) if training else trange(steps)
    # The sampling loop
    for i in generator:
        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            v = model(x, ts * t[i], cond).float()
     
        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < steps - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            x = pred * alphas[i + 1] + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma

    # If we are on the last timestep, output the denoised image
    return pred



class DiffusionUncond(nn.Module):
    def __init__(self,                                  #per hugo 6, per noi 3
                 model = DiffusionAttnUnet1D(io_channels=6, n_attn_layers=4) , 
                 seed = 42,
                 ema_decay = 0.995):
        super().__init__()

        self.diffusion = model
        self.diffusion_ema = deepcopy(self.diffusion)
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True, seed=seed)
        self.ema_decay = ema_decay
        
  
    def training_step(self, y, fc,lambda_corr, cond = None, device=None):
        reals = y # y,x,*others 

        # Draw uniformly distributed continuous timesteps
        t = self.rng.draw(reals.shape[0])[:, 0].to(device)

        t = get_crash_schedule(t)

        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = get_alphas_sigmas(t)

        # Combine the ground truth images and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(reals)
        noised_reals = reals * alphas + noise * sigmas
        targets = noise * alphas - reals * sigmas
        v = self.diffusion(noised_reals, t, cond)
        
        # Adding the MSE and the interfrequency correlation loss to the cost function
        lossMSE = F.mse_loss(v, targets)
        loss_f =  Loss_Covariant(v, alphas, sigmas, noised_reals, fc, device)
        loss =  lossMSE + lambda_corr*loss_f
        print(f"MSE loss: {lossMSE}")
        print(f"Frequency loss: {loss_f}")
        print(f"Total Loss: {loss}")
        print(f"lambda_corr: {lambda_corr}")
        
        return loss
    
    def sample(self, x, num_steps, device = torch.device("cpu"), training = True):
        noise = torch.randn_like(x).to(device)
        out = sample(model = self.diffusion,
                     noise = noise,
                     cond = x,
                     steps = num_steps,
                     eta = 0,
                     training = training,
                     device = device)
        return out

    def test_step(self,y,x,num_steps,device):
        xt = self.sample(x, num_steps, device)
        return {
            "MSE": MSE(y,xt),
            "SNR" : snr(y,xt),
            "lsd" : log_spectral_distance(y,xt)
        }
    
if __name__ == "__main__":
    from diffusion.diffusion_model import DiffusionAttnUnet1DCond, DiffusionAttnUnet1D
    from torchsummary import summary
    model = DiffusionAttnUnet1DCond(io_channels=6,n_attn_layers=4,depth=4,c_mults=[128, 128, 256, 256] + [512] )
    #summary(model)
    #state_dict = torch.load("/content/my_trained_model_.pth", map_location=torch.device('cpu'))
    #model= model.load_state_dict(state_dict)
    #model.eval()
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DiffusionUncond(model)
    #x = torch.randn(5,3,6000)
    x= torch.randn(1,3,6000)
    reals = x
    # Set device (CPU or GPU)
    device = torch.device("cpu")
    x = x.to(device)
    # Generate output using the model
    num_steps = 2  # Adjust based on desired sampling steps
    output = model.sample(x, num_steps, device=device, training=True)

    # Plot and save the model's output
    plt.plot(output[0, 0, :].cpu().detach().numpy())  # Convert to NumPy for plotting
    plt.xlabel("Sample Index")
    plt.ylabel("Generated Signal Amplitude")
    plt.title("Generated Signal from Diffusion Model")
    plt.grid(True)
    save_path = "/content/noisy_signal.png"
    plt.savefig(save_path)
    plt.show()

    import numpy as np
    fs=6000
    T=1/fs
    N = output.shape[2]  
    fft_values = np.fft.fft(output[0,0,:])  # FFT del segnale
    frequencies = np.fft.fftfreq(N, T)  # Calcolo delle frequenze
    half_N = N // 2
    fft_magnitude = np.abs(fft_values[:half_N])  # Modulo della FFT
    frequencies = frequencies[:half_N]  # Frequenze corrispondenti
    plt.subplot(2, 1, 2)
    plt.plot(frequencies, fft_magnitude, 'r')
    plt.title("Spettro del segnale (dominio delle frequenze)")
    plt.xlabel("Frequenza [Hz]")
    plt.ylabel("Ampiezza")
    plt.grid() 
    save_path = "/content/noisy_signal2.png"
    plt.savefig(save_path)
    plt.show() 

    

