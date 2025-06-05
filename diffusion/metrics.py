import numpy as np
from tqdm import tqdm
import math
import torch
import torch.nn as nn
from obspy.signal.tf_misfit import eg, pg,plot_tf_gofs
from numpy import linalg
from torchmetrics.audio import SignalNoiseRatio


def snr(pred, target,eps=0):
    noise = target - pred

    snr_value = (torch.sum(target**2, dim=-1) + eps) / (torch.sum(noise**2, dim=-1) + eps)
    snr_value = 10 * torch.log10(snr_value)
    """
    return (20 *torch.log10(torch.norm(target, dim=-1) \
            /torch.norm(pred -target, dim =-1).clamp(min =1e-8))).mean()
    """
    return snr_value.mean()

def MSE(pred, target):
    loss = nn.MSELoss()
    loss_v = loss(pred, target)
    return loss_v

def gof(pred,target):
    eg_value = eg(pred,target)
    pg_value = pg(pred,target)
    return eg_value, pg_value

def compute_embeddings(classifier_model,dataloader, count):
    image_embeddings = []


    for _ in tqdm(range(count)):
        images = next(iter(dataloader))
        embeddings = classifier_model.predict(images)
        image_embeddings.extend(embeddings)

    return np.array(image_embeddings)




def calculate_fid(real_embeddings, generated_embeddings):
    # calculate mean and covariance statistics
    mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
    mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(generated_embeddings,  rowvar=False)
     # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
       covmean = covmean.real
     # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid



def log_spectral_distance(x1, x2):

    # Compute the LSD
    difference = torch.log(x1) - torch.log(x2)
    lsd_value = torch.sqrt(torch.mean(difference ** 2, dim=1))
    
    # Return the average LSD across the three rows (assuming they represent 3 different spectra)
    return lsd_value.mean()