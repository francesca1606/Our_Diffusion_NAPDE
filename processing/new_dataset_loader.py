import torch
import random
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import matplotlib.pyplot as plt


class AugmentedDatasetSTEAD(Dataset):
    """
    Dataset for the STEAD data, which contains ground acceleration data (torch.tensor) and information
    about the corner frequency (fc).

    Args:
        path (str): Path to the STEAD data directory.
        predict_pga (bool): If True, the dataset will return the true PGA values instead of normalized broadband data.
        data_percent (float): Percentage of the dataset to use (default is 1, meaning 100%).

    """

    def __init__(self,
                 path="STEAD_data/chunk2/",
                 predict_pga=False,
                 data_percent=1,
                 *args, **kwargs
                 ):
        super(AugmentedDatasetSTEAD, self).__init__()

        self.path = path
        self.predict_pga = predict_pga
        self.broadband_dataset = torch.load(
            self.path + "chunk2_130k_acceleration_no_nan.pt")
        self.fc = torch.load(self.path + "chunk2_130k_fc_no_nan.pt")
        # self.meta_data_depth           = torch.load(self.path + "meta_data_depth.pt")   # NON SO SE LI VOGLIAMO METTERE
        # self.meta_data_magnitude       = torch.load(self.path + "meta_data_magnitude.pt")
        self.data_percent = data_percent

    def __len__(self):
        return int(self.data_percent*len(self.broadband_dataset))

    def __getitem__(self, index):
        broadband = self.broadband_dataset[index, :, :].float()
        fc = self.fc[index, None].float()
        # depth       = self.meta_data_depth[index].float()
        # magnitude   = self.meta_data_magnitude[index].float()

        true_pga, _ = torch.max(torch.abs(broadband), dim=1)
        # Normalize broadband and lowpass
        if not self.predict_pga:
            broadband = broadband / (true_pga[:, None] + 1e-8)
        return broadband, fc, true_pga  # depth, magnitude


class AugmentedDataModule:
    def __init__(self,
                 batch_size=64,
                 num_workers=4,
                 path="STEAD_data/chunk2/",
                 predict_pga=False,
                 seed=42,
                 *args, **kwargs
                 ) -> None:

        self.ds = AugmentedDatasetSTEAD(path, predict_pga=predict_pga)

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
        train_partition, valid_partition = int(
            0.80*len(self.ds)), int(0.10*len(self.ds))
        test_partition = len(self.ds) - train_partition - valid_partition
        generator = torch.Generator().manual_seed(self.seed)
        self.train_ds, self.valid_ds, self.test_ds = torch.utils.data.random_split(
            self.ds, [train_partition, valid_partition, test_partition], generator=generator)

    def define_dataloader(self):

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True

        )
        self.test_loader = DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

        self.valid_loader = DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
        self.combined_loader = DataLoader(
            self.ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True
        )

        y, x, *other = next(iter(self.train_loader))
        self.one_batch = DataLoader(TensorDataset(y, x, *other),
                                    batch_size=1)

    def setup(self):
        self.define_dataset()
        self.define_dataloader()


class AugmentedDatasetPBS(Dataset):
    """
    Dataset for the data from physics based simulation, which contains low frequency ground acceleration data
    (torch.tensor). The dataset can be configured to use a percentage of the total data.

    Args:
        path (str): Path to the PBS data directory.
        data_percent (float): Percentage of the dataset to use (default is 1, meaning 100%).
    """

    def __init__(self,
                 path="path_to_pbs_data/",
                 data_percent=1,
                 *args, **kwargs
                 ):
        super(AugmentedDatasetPBS, self).__init__()

        self.path = path
        self.pbs_lowfreq_dataset = torch.load(self.path + "pbs_data.pt")
        self.data_percent = data_percent

    def __len__(self):
        return int(self.data_percent*len(self.pbs_lowfreq_dataset))

    def __getitem__(self, index):
        pbs_lowfreq = self.pbs_lowfreq_dataset[index, :, :].float()
        return pbs_lowfreq


class AugmentedDataModulePBS:
    """
    Data module for the PBS dataset.
    """

    def __init__(self,
                 batch_size=64,
                 num_workers=4,
                 path="path_to_pbs_data/",
                 seed=42,
                 *args, **kwargs
                 ) -> None:

        self.ds = AugmentedDatasetPBS(path)

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
        train_partition, valid_partition = int(
            0.80*len(self.ds)), int(0.10*len(self.ds))
        test_partition = len(self.ds) - train_partition - valid_partition
        generator = torch.Generator().manual_seed(self.seed)
        self.train_ds, self.valid_ds, self.test_ds = torch.utils.data.random_split(
            self.ds, [train_partition, valid_partition, test_partition], generator=generator)

    def define_dataloader(self):

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True

        )
        self.test_loader = DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

        self.valid_loader = DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
        self.combined_loader = DataLoader(
            self.ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True
        )

        x = next(iter(self.train_loader))
        self.one_batch = DataLoader(TensorDataset(x), batch_size=1)

    def setup(self):
        self.define_dataset()
        self.define_dataloader()
