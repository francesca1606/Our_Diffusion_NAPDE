import torch
import random
from torch.utils.data import DataLoader,TensorDataset 
import numpy as np
from diffusion.visualisation import visualise_fft
from diffusion.visualisation import visualize_signal
import matplotlib.pyplot as plt
from diffusion.duo_graph import amplitude_graph, frequency_loglog

class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self,
                  path ="data/nsy12800",
                    *args, **kwargs):
        super(AugmentedDataset, self).__init__()
        
        self.path = path
        self.broadband_dataset         = torch.load(self.path + "broadband.pt")
        self.lowpass_filter_dataset    = torch.load(self.path + "lowpass.pt")
        self.meta_data_depth           = torch.load(self.path + "meta_data_depth.pt")
        self.meta_data_magnitude       = torch.load(self.path + "meta_data_magnitude.pt")
        self.pga_broadband             = torch.load(self.path + "pga_broadband.pt")

    def __len__(self): 
        return len(self.broadband_dataset)

    def __getitem__(self,index):
        broadband   = self.broadband_dataset[index,:,:].float()
        lowpass     = self.lowpass_filter_dataset[index,:,:].float()
        depth       = self.meta_data_depth[index].float()
        magnitude   = self.meta_data_magnitude[index].float()
        pga         = self.pga_broadband[index,:].float()
    
        return broadband, lowpass, pga, depth, magnitude


class AugmentedDataset6000(torch.utils.data.Dataset):
    def __init__(self,
                  path ="data/6000_data/",
                  predict_pga = False,
                  data_percent = 1,
                    *args, **kwargs
                  ):
        super(AugmentedDataset6000, self).__init__()
        
        self.path = path
        self.predict_pga = predict_pga
        print("inside the function")
        self.broadband_dataset         = torch.load("./data_float16/bb_16.pt")
        print("broadband dataset loaded")
        self.lowpass_filter_dataset    = torch.load("./data_float16/lp_16.pt")
        print("lowpass_filter dataset loaded")
        self.meta_data_depth           = torch.load(self.path + "meta_data_depth.pt")
        print("meta data depth loaded")
        self.meta_data_magnitude       = torch.load(self.path + "meta_data_magnitude.pt")
        print("meta data magnitude loaded")
        self.pga_broadband             = torch.load(self.path + "pga_broadband.pt")
        print("pga_broadband loaded")
        self.data_percent = data_percent

    def __len__(self): 
        return int(self.data_percent*len(self.broadband_dataset))

    def __getitem__(self,index):
        broadband   = self.broadband_dataset[index,:,:].float()
        lowpass     = self.lowpass_filter_dataset[index,:,:].float()
        depth       = self.meta_data_depth[index].float()
        magnitude   = self.meta_data_magnitude[index].float()
        pga         = self.pga_broadband[index,:].float() # The PGA in the original datast wasn't accurate, so we don't use it
        
        true_pga,_ = torch.max(torch.abs(broadband),dim = 1)
        max_lowpass,_ = torch.max(torch.abs(lowpass),dim = 1)
        # Normalize broadband and lowpass
        if not self.predict_pga:
            broadband = broadband / true_pga[:,None]
            lowpass = lowpass / max_lowpass[:,None]
        return broadband, lowpass, true_pga, max_lowpass, depth, magnitude



class AugmentedDataModule : 
    def __init__(self,
                 batch_size  = 64,
                 num_workers = 4,
                 path = "data/6000_data/",
                 predict_pga = False,
                 seed = 42,
                 data_percent = 1,
                 *args, **kwargs
                 ) -> None:
        
        if path == "data/nsy51200/temporary/":
            self.ds = AugmentedDataset(path)
        else : 
            self.ds = AugmentedDataset6000(path, predict_pga = predict_pga)
        
        """
        elif : path == "data/6000_data/":
            self.ds = AugmentedDataset6000(path, predict_pga = predict_pga)
        """    

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.set_seed(seed)

    def set_seed(self, seed_value=42):
        """Set seed for reproducibility."""
        random.seed(seed_value)  # Python random module
        np.random.seed(seed_value)  # Numpy module
        torch.manual_seed(seed_value)  # PyTorch
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups

    def define_dataset(self):
        train_partition, valid_partition = int(0.80*len(self.ds)), int(0.10*len(self.ds))
        test_partition = len(self.ds) - train_partition - valid_partition
        generator = torch.Generator().manual_seed(self.seed)
        self.train_ds, self.valid_ds, self.test_ds = torch.utils.data.random_split(
            self.ds, [train_partition, valid_partition, test_partition], generator=generator)


    def define_dataloader(self): 
        
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size = self.batch_size,
            num_workers = self.num_workers

        )
        self.test_loader = DataLoader(
            self.test_ds,
            batch_size = self.batch_size,
            num_workers = self.num_workers
        )

        self.valid_loader = DataLoader(
            self.valid_ds,
            batch_size = self.batch_size,
            num_workers = self.num_workers
        )
        self.combined_loader = DataLoader(
            self.ds,
            batch_size = self.batch_size,
            num_workers = self.num_workers
        )

        y,x,*other = next(iter(self.train_loader))
        self.one_batch = DataLoader(TensorDataset(y,x,*other),
                                    batch_size= 1  )
    
    def setup(self):
        self.define_dataset()
        self.define_dataloader()
    
 
        
if __name__ == "__main__":
    path = "data/6000_data/"
    module = AugmentedDataModule(path=path, batch_size=20, predict_pga=False,data_percent=0.1)
    module.setup()
    for idx, batch in enumerate(module.combined_loader):
        y,x, pga,max_x,*other = batch
        print(x.shape,y.shape,pga.shape)
        """
        for idx2,(d,y1,maxx1 ) in enumerate(zip(pga,y,max_x)):
            print(f"pga : {d}")
            print(f"max_x : {maxx1}")
            print()
        """
        break 
    """
    histogram_PGA(module.train_loader)
    print("Histogram saved !")
    """