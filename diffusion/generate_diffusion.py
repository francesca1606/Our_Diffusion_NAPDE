import torch
from sklearn.metrics import mean_squared_error
import torch
import matplotlib.pyplot as plt
import os
from diffusion.visualisation import visualize_signal
from diffusion.visualisation import save_frequency_graph, visualisation_3d, save_phase, save_amplitude_graph, save_frequency_graph_loglog
# from processing.augmented_dataset import AugmentedDataModule
from processing.new_dataset_loader import AugmentedDataModule, AugmentedDataModulePBS
from processing.load_data import DataModule
from common.common_nn import patch
from diffusion.metrics import MSE, snr
from diffusion.diff import DiffusionUncond
from diffusion.diffusion_model import DiffusionAttnUnet1DCond, DiffusionAttnUnet1D
# from diffusion.duo_graph import amplitude_graph, frequency_loglog
from diffusion.duo_graph import amplitude_graph_STEAD, frequency_loglog_STEAD, frequency_loglog_pbs, amplitude_graph_pbs
from diffusion.utils import butterworth_decompose_batch
from diffusion.metrics import ssim_skimage, snr
import xgboost as xgb
DEMO_STEPS = 1000
BATCH_SIZE = 4


def load_model(
    path="trained_models/4e_cond.pth"
):
    model = DiffusionAttnUnet1DCond(
        io_channels=3,
        n_attn_layers=4,
        depth=4,
        c_mults=[128, 128, 256, 256] + [512]
    )
    model = DiffusionUncond(model=model)
    device = torch.device(
        "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def load_model_cond(
    path="checkpoint_passati/diffusion_model_Hugo.pth"
):
    model = DiffusionAttnUnet1DCond(
        io_channels=6,
        latent_dim=0,
        n_attn_layers=4,
        depth=4,
        c_mults=[128, 128, 256, 256] + [512]
    )
    model = DiffusionUncond(model=model)
    device = torch.device(
        "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def generate_sample_STEAD():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = load_model()
    model = model.to(device)
    # model = torch.compile(model)
    print("Model loaded")
    path = "generated_test_epoch1/4e_cond"
    i = 0
    while os.path.exists(path[:-1] + str(i)):
        i += 1
    path = path[:-1] + str(i)
    os.makedirs(path)

    dataset = AugmentedDataModule(
        batch_size=BATCH_SIZE,
        path="STEAD_data/chunk2/",
        predict_pga=False,
        data_percent=0.1
    )
    '''
    dataset.setup()
    dataloader = dataset.test_loader
    print("Dataset loaded")
    print("Starting generation")
    for idx,batch in enumerate(dataloader):
        y,*other = batch
        y = y.to(device)
        print(f"y shape BEFORE patch: {y.shape}")
        y, *other= patch(y)
        print(f"y shape AFTER patch: {y.shape}")
        generate = model.sample(y,num_steps = 10,device= device, training=False)
        # Run though the batch
        
        for idx,(y_1,generate_1,) in enumerate(zip(y,generate)):
            amplitude_graph_STEAD(y_1, generate_1,path, idx)
            frequency_loglog_STEAD(y_1, generate_1,path, idx)
    '''

    dataset.setup()
    dataloader = dataset.test_loader
    print("Dataset loaded")
    print("Starting generation")
    for idx, batch in enumerate(dataloader):
        y, *other = batch
        y = y.to(device)
        print(y.shape)
        generate = model.sample(x=None, num_steps=100,
                                device=device, training=False)
        for idx_1, (y_1, generate_1) in enumerate(zip(y, generate)):
            amplitude_graph_STEAD(y_1, generate_1, path, idx + idx_1)
            frequency_loglog_STEAD(y_1, generate_1, path, idx + idx_1)
            print(f"SNR: {snr(y_1, generate_1)}")
            print(f"SSIM: {ssim_skimage(y_1, generate_1)}")


def generate_sample_STEAD_cond():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = load_model_cond()
    model = model.to(device)
    # model = torch.compile(model)
    print("Model loaded")
    path = "generated_test_epoch1/4e_cond"
    i = 0
    while os.path.exists(path[:-1] + str(i)):
        i += 1
    path = path[:-1] + str(i)
    os.makedirs(path)

    dataset = AugmentedDataModule(
        batch_size=BATCH_SIZE,
        path="path_to_pbs_data/",
        predict_pga=False,
        data_percent=0.1
    )

    dataset.setup()
    dataloader = dataset.test_loader
    print("Dataset loaded")
    print("Starting generation")
    for idx, batch in enumerate(dataloader):
        y, *other = batch
        y = y.to(device)
        print(y.shape)
        cond_low = butterworth_decompose_batch(y)
        generate = model.sample(x=cond_low, num_steps=100,
                                device=device, training=False)
        for idx_1, (y_1, generate_1) in enumerate(zip(y, generate)):
            amplitude_graph_STEAD(y_1, generate_1, path, idx + idx_1)
            frequency_loglog_STEAD(y_1, generate_1, path, idx + idx_1)


def generate_sample_pbs():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = load_model_cond()
    model = model.to(device)
    # model = torch.compile(model)
    print("Model loaded")
    path = "generated_test_epoch1/4e_cond"
    i = 0
    while os.path.exists(path[:-1] + str(i)):
        i += 1
    path = path[:-1] + str(i)
    os.makedirs(path)

    dataset = AugmentedDataModulePBS(
        batch_size=BATCH_SIZE,
        path="STEAD_data/chunk2/",
        data_percent=0.1
    )

    dataset.setup()
    dataloader = dataset.test_loader
    print("Dataset loaded")
    print("Starting generation")
    for idx, batch in enumerate(dataloader):
        x = batch
        x = x.to(device)
        print(x.shape)
        generate = model.sample(x=x, num_steps=1000,
                                device=device, training=False)
        for idx_1, (x_1, generate_1) in enumerate(zip(x, generate)):
            amplitude_graph_pbs(x_1, path, idx + idx_1)
            frequency_loglog_pbs(x_1, path, idx + idx_1)


if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    print(f"Gpu available : {cuda}")
    generate_sample_STEAD_cond()


def main():
    cuda = torch.cuda.is_available()
    print(f"Gpu available : {cuda}")
    generate_sample_STEAD_cond()
