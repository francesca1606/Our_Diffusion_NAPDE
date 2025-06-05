import torch
from sklearn.metrics import mean_squared_error
import torch
import matplotlib.pyplot as plt
import os
from diffusion.visualisation import visualize_signal
from diffusion.visualisation import save_frequency_graph, visualisation_3d, save_phase, save_amplitude_graph, save_frequency_graph_loglog
#from processing.augmented_dataset import AugmentedDataModule
from processing.new_dataset_loader import AugmentedDataModule
from processing.load_data import DataModule
from common.common_nn import patch
from diffusion.metrics import MSE,snr
from diffusion.diff import DiffusionUncond
from diffusion.diffusion_model import DiffusionAttnUnet1DCond
#from duo_graph import amplitude_graph, frequency_loglog
from diffusion.duo_graph import amplitude_graph_STEAD, frequency_loglog_STEAD
import xgboost as xgb
DEMO_STEPS = 1000
BATCH_SIZE = 4


def load_model(
        #path = "checkpoints/diffusion_model.pth",
        path = "trained_models/my_trained_model_2_06_lr_1e-3_epoch_6.pth"
    ):
    model = DiffusionAttnUnet1DCond(
                    io_channels=3, #6
                    n_attn_layers=4,
                    depth=4,
                    c_mults=[128, 128, 256, 256] + [512] 
        )
    model = DiffusionUncond(model = model)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model.load_state_dict(torch.load(path, map_location = device))
    model.eval()
    return model


def generate_sample():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = load_model()
    model = model.to(device)
    #model = torch.compile(model)
    print("Model loaded")
    path = "generated_test_epoch6/experiment_0"
    i = 0
    while os.path.exists(path[:-1] + str(i)):
        i += 1
    path = path[:-1] + str(i)
    os.makedirs(path)
    
    dataset = AugmentedDataModule(
        batch_size = BATCH_SIZE,
        path="STEAD_data/chunk2/",   #"data/6000_data/",
        predict_pga= False,
        data_percent=0.1
    )
    dataset.setup()
    dataloader = dataset.test_loader
    print("Dataset loaded")
    print("Starting generation")
    for idx,batch in enumerate(dataloader):
        y,x,*other = batch
        y = y.to(device)
        x = x.to(device)
        #print(f"y shape BEFORE patch: {y.shape}")
        y, x = patch(y, x)
        #print(f"x shape AFTER patch: {x.shape}")
        #if x.shape[1] < 19:
        #  pad_channels = 19 - x.shape[1]
        #  x = torch.nn.functional.pad(x, (0, 0, 0, pad_channels))  # Pad sui canali
        #print(f"x shape after padding: {x.shape}")
        input = torch.randn(x.shape[0], 3, 6000).to(device)  #AGGIUNTO
        generate = model.sample(x,num_steps = 10,device= device, training=False)
        # Run though the batch
        
        for idx,(x_1,y_1,generate_1,) in enumerate(zip(x,y,generate)):
            amplitude_graph(y_1, generate_1,x_1 ,path, idx)
            frequency_loglog(y_1, generate_1,x_1 ,path, idx)
            
            
def generate_sample_STEAD():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = load_model()
    model = model.to(device)
    #model = torch.compile(model)
    print("Model loaded")
    path = "generated_test_epoch6/experiment_0"
    i = 0
    while os.path.exists(path[:-1] + str(i)):
        i += 1
    path = path[:-1] + str(i)
    os.makedirs(path)
    
    dataset = AugmentedDataModule(
        batch_size = BATCH_SIZE,
        path="STEAD_data/chunk2/",   #"data/6000_data/",
        predict_pga= False,
        data_percent=0.1
    )
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

def generate_sample_with_pga():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = load_model()
    model = model.to(device)
    model = torch.compile(model)
    path = "generated_test_epoch6/diffusion/experiment_0"
    i = 0
    while os.path.exists(path[:-1] + str(i)):
        i += 1
    path = path[:-1] + str(i)
    os.makedirs(path)
    
    dataset = AugmentedDataModule(
        batch_size = BATCH_SIZE,
        path="STEAD_data/chunk2/", #"data/6000_data/",
        predict_pga= True,
        data_percent=0.1
    )
    dataset.setup()
    dataloader = dataset.test_loader
    bst = xgb.Booster({'gpu_id': 0})
    bst.load_model("models/XGBoostPGA/final.model")
    for idx,batch in enumerate(dataloader):
        y,x, pga, max_lowpass,*other = batch
        y = y.to(device)
        x = x.to(device)
        pga = pga.to(device)
        max_lowpass = max_lowpass.to(device)
        #not_normalized_x = x * max_lowpass
        #not_normalized_y = y * pga
        xgdbmatrix = xgb.DMatrix(x.detach().cpu().numpy().reshape(x.shape[0],-1))
        predicted_pga = bst.predict(xgdbmatrix)
        predicted_pga = torch.Tensor(predicted_pga).unsqueeze(2).to(device)
        """ for pga1,predicted_pga1 in zip(pga,predicted_pga):
            pga1 = pga1.detach().cpu().numpy()
            print(mean_squared_error(pga1,predicted_pga1))
        """
        y,x = patch(y,x)
        generate = model.sample(x,num_steps = 1000, device= device, training=False)
        # Run though the batch
        print(f"generate : {generate.shape}")
        print(f"pga : {pga.shape}")
        print(f"predicted pga : {predicted_pga.shape}")
        print(f"y : {y.shape}") 
        y = y * pga.unsqueeze(2)
        x = x * max_lowpass.unsqueeze(2)
        generate = generate * predicted_pga
        
        for idx,(x_1,y_1,generate_1,pga1,predicted_pga1) in enumerate(zip(x,y,generate,pga,predicted_pga)):
            amplitude_graph(y_1, generate_1,x_1 ,path, idx)
            frequency_loglog(y_1, generate_1,x_1 ,path, idx)
        break



if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    print(f"Gpu available : {cuda}")
    generate_sample_STEAD()

def main():
    cuda = torch.cuda.is_available()
    print(f"Gpu available : {cuda}")
    generate_sample_STEAD()

