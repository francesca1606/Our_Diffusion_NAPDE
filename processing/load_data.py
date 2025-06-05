import torch
from torch.utils.data import DataLoader, ConcatDataset,TensorDataset 
from diffusion.visualisation import visualize_signal,visualise_fft, visualize_low_high
from common.common_nn import patch
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
from diffusion.duo_graph import amplitude_graph, frequency_loglog

class DataModule:
    def __init__(self,
        path = "data/nsy12800/",
        sample_rate = 30,
        conditional = True,
        batch_size = 64,
        num_workers = 4,
        shuffle = True
    ):

        self.path = path
        self.sample_rate = sample_rate
        self.conditional = conditional
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def load_pth(self):
        self.train_pth = torch.load(self.path + "ths_trn_nt4096_ls128_nzf8_nzd32.pth")
        self.test_pth = torch.load(self.path + "ths_tst_nt4096_ls128_nzf8_nzd32.pth")
        self.valid_pth = torch.load(self.path + "ths_vld_nt4096_ls128_nzf8_nzd32.pth")

    def define_dataloader(self):

        self.train_loader = DataLoader(
            self.train_pth, 
            batch_size = self.batch_size,
            shuffle = self.shuffle,
            num_workers = self.num_workers,
       )

        self.test_loader = DataLoader(
            self.test_pth, 
            batch_size = self.batch_size,
            shuffle = self.shuffle,
            num_workers = self.num_workers
        )

        self.valid_loader = DataLoader(
            self.valid_pth, 
            batch_size = self.batch_size,
            shuffle = self.shuffle,
            num_workers = self.num_workers
        )
        self.combined_loader = DataLoader(
            ConcatDataset([self.train_pth, self.test_pth, self.valid_pth]),
            batch_size = self.batch_size,
            shuffle = self.shuffle,
            num_workers = self.num_workers
        )
        y,x,*other = next(iter(self.train_loader))

        self.one_batch = DataLoader(TensorDataset(y,x,*other),
                                    batch_size= 1  )

    def setup(self):
        self.load_pth()
        self.define_dataloader()


if __name__ == "__main__":
    module = DataModule(shuffle= True, batch_size=32)
    module.setup()
    for idx, batch in enumerate(module.combined_loader):
        y,x,pga ,*other = batch
        print(x.shape,pga.shape)
        """         
        x_0, y_0 = x[0], y[0]
        amplitude_graph(y_0, generate= x_0,x = x_0, path = "test", idx = idx)
        frequency_loglog(y_0, generate= x_0,x = x_0, path = "test", idx = idx)
        if idx > 0:
            break """