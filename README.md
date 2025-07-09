# STEAD_Diffusion

## Credits
This repository is a variation of "https://github.com/HugoGabrielidis16/STEAD_Diffusion", developed by Hugo Gabrielidis, Université Paris-Saclay, CentraleSupélec, CNRS, ENS; Laboratoire de Mécanique Paris-Saclay UMR 9026

## Dataset
You can download the dataset and the model checkpoints at the following link : [https://drive.google.com/drive/folders/1LSCA4RlH3TpagfEpWpyQS7ABPFQgwpoK?usp=drive_link](https://drive.google.com/drive/u/2/folders/1NDLndXlhavJR8IElo1Fos1ouWshuPnGG)

The files should be placed as follow : <br />
    - STEAD_DATA/chunk2/chunk2_130k_fc_no_nan.pt <br />
    - STEAD_DATA/chunk2/chunk2_130k_acceleration_no_nan.pt <br />
    - checkpoints/diffusion_model_cond.pth <br />
    - checkpoints/diffusion_model_uncond.pth <br />

## Instructions to run
After cloning the repository, run the following command inside the repository folder to be able to execute all python scripts:

    export PYTHONPATH=$(pwd)
    
    pip install -r requirements.txt

To execute a training of the model, change the model path, number of epochs, batch size, conditioning,... in 'parsing/parsing.py'

    python training/train.py

To generate synthetic seismic signals starting from the true ones, run

    python diffusion/generate_diffusion.py


    
