## STEAD DATA PROCESSING 

In this folder you can find find three notebooks that can be used to process the data taken from the [STEAD Dataset](https://github.com/smousavi05/STEAD) in such a way that can be interfaced with the model. 

In particular the processing of the data was devided in three steps: 
1. `hdf5_to_torch.ipynb`: converts chunk2 STEAD data to pytorch
2. `calculating_fc_for_dataset.ipynb`: calculates for each signal the corresponding corner frequency fc
3. `remove_NaN.ipynb`: removes coherently the eventual NaNs from `fc.pt` and `chunk_i_acceleration.pt`

An additional notebook, `trimmed_model_uncond_diffusion.ipynb`, adapts the original checkpoint of AttUnet1DCond to the case in which no conditioning is performed

