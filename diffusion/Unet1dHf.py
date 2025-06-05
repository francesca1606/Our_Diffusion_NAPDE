import torch
import torch.nn as nn
from diffusers import UNet1DModel
from torchsummary import summary


class UNET_1d(UNet1DModel):
    def __ini__(*args,**kwargs):
        super().__init__(*args,**kwargs)


    def forward(self, sample, timestep, condition, return_dict = False):
        super().forward(sample, timestep, return_dict = return_dict)[0]
    


class UNet1dHF(nn.Module):                                                    
    def __init__(self, in_channels = 3, out_channels = 3 , extra_in_channels = 19   ):
        super().__init__()
        self.model = UNet1DModel(
            sample_size=4096,
            in_channels=in_channels,
            out_channels=out_channels,
            extra_in_channels= extra_in_channels,
            layers_per_block=20,
            block_out_channels=(32,32,64,64,128,128,512),
            down_block_types=(
                "DownBlock1DNoSkip",
                "DownBlock1D", 
                "AttnDownBlock1D",
                "DownBlock1D", 
                "AttnDownBlock1D",
                "DownBlock1D", 
                "AttnDownBlock1D"
            ),
            up_block_types=(
                "AttnUpBlock1D", 
                "UpBlock1D", 
                "AttnUpBlock1D", 
                "UpBlock1D", 
                "AttnUpBlock1D", 
                "UpBlock1D", 
                "UpBlock1DNoSkip"
       
            ),  
            mid_block_type = "UNetMidBlock1D",
        )
        
    def forward(self, sample, timestep, condition, return_dict = False):
        sample = torch.cat([sample, condition], dim=1)
        return self.model(sample, timestep, return_dict = return_dict)[0]

if __name__ == "__main__":
    model = UNet1dHF(0,3,19)  #3,3
    bs = 13
    x = torch.rand([bs,3,4096])
    cond = torch.rand([bs,3,4096])
    timestep = torch.rand([bs])
    y = model(x,timestep, cond)
    print(y.shape)
    summary(model)
