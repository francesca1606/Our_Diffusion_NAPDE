import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary
from diffusion.blocks import SkipBlock, FourierFeatures, SelfAttention1d, ResConvBlock, Downsample1d, Upsample1d, SequentialY, DualInputSequential, CoreBlock, CrossAttention1d, CrossAttentionSkipBlockY, HalfChannels1d, Upsample1dHalfChannels
from diffusion.utils import append_dims, expand_to_planes
from diffusion.utils import wavelet_decompose_batch, butterworth_decompose_batch

class DiffusionAttnUnet1D(nn.Module):
    def __init__(
        self, 
        io_channels = 3, 
        latent_dim = 0,
        depth=11, # diminuated the depth make it work check highest value (11 is the highest value possible here)
        n_attn_layers = 6,
        #c_mults = [64,64,128,128] + [256]*10

        c_mults = [128, 128, 256, 256] + [512] * 10
        
    ):
        super().__init__()

        self.timestep_embed = FourierFeatures(1, 16)

        self.latent_dim = latent_dim

        attn_layer = depth - n_attn_layers - 1

        block = nn.Identity()

        conv_block = ResConvBlock

        for i in range(depth, 0, -1):
            c = c_mults[i - 1]
            if i > 1:
                c_prev = c_mults[i - 2]
                add_attn = i >= attn_layer and n_attn_layers > 0
                block = SkipBlock(
                    Downsample1d("cubic"),
                    conv_block(c_prev, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    block,
                    conv_block(c * 2 if i != depth else c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c_prev),
                    SelfAttention1d(c_prev, c_prev //
                                    32) if add_attn else nn.Identity(),
                    Upsample1d(kernel="cubic")
                    # nn.Upsample(scale_factor=2, mode='linear',
                    #             align_corners=False),
                )
            else:
                block = nn.Sequential(
                    conv_block(io_channels + 16 + self.latent_dim, c, c),
                    conv_block(c, c, c),
                    conv_block(c, c, c),
                    block,
                    conv_block(c * 2, c, c),
                    conv_block(c, c, c),
                    conv_block(c, c, io_channels, is_last=True),
                )
        self.net = block

        with torch.no_grad():
            for param in self.net.parameters():
                param *= 0.5

    def forward(self, input, t, cond=None):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        inputs = [input, timestep_embed]
        if cond is not None:
            cond = F.interpolate(cond, (input.shape[2], ), mode='linear', align_corners=False)
            inputs.append(cond)

        return self.net(torch.cat(inputs, dim=1))

class DiffusionAttnUnet1DCond(nn.Module):
    def __init__(
        self, 
        io_channels = 3,
        latent_dim = 0,
        depth=11, # diminuated the depth make it work check highest value (11 is the highest value possible here)
        n_attn_layers = 6,
        c_mults = [128, 128, 256, 256] + [512] * 10
        
    ):
        super().__init__()

        self.timestep_embed = FourierFeatures(1, 16)  

        self.latent_dim = latent_dim

        attn_layer = depth - n_attn_layers - 1

        block = nn.Identity()

        conv_block = ResConvBlock

        for i in range(depth, 0, -1):
            c = c_mults[i - 1]
            if i > 1:
                c_prev = c_mults[i - 2]
                add_attn = i >= attn_layer and n_attn_layers > 0
                block = SkipBlock(
                    Downsample1d("cubic"),
                    conv_block(c_prev, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    block,
                    conv_block(c * 2 if i != depth else c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c_prev),
                    SelfAttention1d(c_prev, c_prev //
                                    32) if add_attn else nn.Identity(),
                    Upsample1d(kernel="cubic")
                    # nn.Upsample(scale_factor=2, mode='linear',
                    #             align_corners=False),
                )
            else:
                block = nn.Sequential(
                    conv_block(io_channels + 16 + self.latent_dim, c, c),
                    conv_block(c, c, c),
                    conv_block(c, c, c),
                    block,
                    conv_block(c * 2, c, c),
                    conv_block(c, c, c),
                    conv_block(c, c, io_channels, is_last=True),
                )
        self.net = block
        self.lastconv = nn.Conv1d(io_channels, 3, 3, padding= "same")
        
        with torch.no_grad():
            for param in self.net.parameters():
                param *= 0.5

    def forward(self, input, t, cond=None):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        inputs = [input, timestep_embed]
        #if cond is not None:
        #    print("QUI")
        #    cond = F.interpolate(cond, (input.shape[2], ), mode='linear', align_corners=False)
        #    inputs.append(cond)
        concatenated_input = torch.cat(inputs, dim=1)
        print(f"\nSample shape: {concatenated_input.shape}")
        out = self.net(concatenated_input)
        #out = self.net(torch.cat(inputs, dim=1))
        out = self.lastconv(out)
        return out 
        
        
# ----------------------
# NEW MODEL
# ----------------------


class DiffusionAttnDualBranchUnet1D(nn.Module):
    def __init__(
        self,
        io_channels=3,
        latent_dim=0,
        # diminuated the depth make it work check highest value (11 is the highest value possible here)
        depth=11,
        n_attn_layers=6,
        n_cattn_layers=6,
        # c_mults = [64,64,128,128] + [256]*10

        c_mults=[128, 128, 256, 256] + [512] * 10

    ):
        super().__init__()

        self.timestep_embed = FourierFeatures(1, 16)

        self.latent_dim = latent_dim

        attn_layer = depth - n_attn_layers - 1

        block = CoreBlock()

        conv_block = ResConvBlock

        for i in range(depth, 0, -1):
            c = c_mults[i - 1]
            if i > 1:
                c_prev = c_mults[i - 2]
                add_attn = i >= attn_layer and n_attn_layers > 0

                block = CrossAttentionSkipBlockY(main1_layers = [Downsample1d("cubic"),
                    conv_block(c_prev, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity()], main2_layers = [Downsample1d("cubic"),
                    conv_block(c_prev, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity()], block = DualInputSequential(block,[conv_block(c * 3 if i != depth else c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c_prev),
                    SelfAttention1d(c_prev, c_prev //
                                    32) if add_attn else nn.Identity(),
                    Upsample1d(kernel="cubic")]), channels = c_prev, n_heads = c // 32)

            else:

                block = SequentialY(main1_layers = [conv_block(io_channels + 16 + self.latent_dim, c, c),
                    conv_block(c, c, c),
                    conv_block(c, c, c)], main2_layers = [conv_block(io_channels + 16 + self.latent_dim, c, c),
                    conv_block(c, c, c),
                    conv_block(c, c, c)], block = DualInputSequential(block,[conv_block(c * 3, c, c),
                    conv_block(c, c, c),
                    conv_block(c, c, io_channels, is_last=True)]))

        self.net = block

        with torch.no_grad():
            for param in self.net.parameters():
                param *= 0.5

    def forward(self, input, t, cond = None):

        #cA, cD = wavelet_decompose_batch(input) 
        with torch.no_grad():
            input_low, input_high = butterworth_decompose_batch(input)

        timestep_embed = expand_to_planes(
            self.timestep_embed(t[:, None]), input_low.shape)

        input_low = torch.cat([input_low, timestep_embed],dim=1) # not really sure about this
        input_high = torch.cat([input_high, timestep_embed],dim=1)

        return self.net(input_high,input_low)




#class DiffusionAttnDualBranchUnet1D(nn.Module):
#    def __init__(
#        self,
#        io_channels=3,
#        latent_dim=0,
#        # diminuated the depth make it work check highest value (11 is the highest value possible here)
#        depth=11,
#        n_attn_layers=6,
#        n_cattn_layers=6,
#        # c_mults = [64,64,128,128] + [256]*10
#
#        c_mults=[128, 128, 256, 256] + [512] * 10
#
#    ):
#        super().__init__()
#
#        self.timestep_embed = FourierFeatures(1, 16)
#
#        self.latent_dim = latent_dim
#
#        attn_layer = depth - n_attn_layers - 1
#
#        block = CoreBlock()
#
#        conv_block = ResConvBlock
#
#        for i in range(depth, 0, -1):
#            c = c_mults[i - 1]
#            if i > 1:
#                c_prev = c_mults[i - 2]
#                add_attn = i >= attn_layer and n_attn_layers > 0
#
#                block = CrossAttentionSkipBlockY(main1_layers = [Downsample1d("cubic"),
#                    conv_block(c_prev, c, c),
#                    SelfAttention1d(
#                        c, c // 32) if add_attn else nn.Identity(),
#                    conv_block(c, c, c),
#                    SelfAttention1d(
#                        c, c // 32) if add_attn else nn.Identity(),
#                    conv_block(c, c, c),
#                    SelfAttention1d(
#                        c, c // 32) if add_attn else nn.Identity()], main2_layers = [Downsample1d("cubic"),
#                    conv_block(c_prev, c, c),
#                    SelfAttention1d(
#                        c, c // 32) if add_attn else nn.Identity(),
#                    conv_block(c, c, c),
#                    SelfAttention1d(
#                        c, c // 32) if add_attn else nn.Identity(),
#                    conv_block(c, c, c),
#                    SelfAttention1d(
#                        c, c // 32) if add_attn else nn.Identity()], block = DualInputSequential(block,[conv_block(c * 3 if i != depth else c, c, c),
#                    SelfAttention1d(
#                        c, c // 32) if add_attn else nn.Identity(),
#                    conv_block(c, c, c),
#                    SelfAttention1d(
#                        c, c // 32) if add_attn else nn.Identity(),
#                    conv_block(c, c, c_prev),
#                    SelfAttention1d(c_prev, c_prev //
#                                    32) if add_attn else nn.Identity(),
#                    Upsample1d(kernel="cubic")]), channels = c_prev, n_heads = 2)
#
#            else:
#
#                block = SequentialY(main1_layers = [conv_block(io_channels + 16 + self.latent_dim, c, c),
#                    conv_block(c, c, c),
#                    conv_block(c, c, c)], main2_layers = [conv_block(io_channels + 16 + self.latent_dim, c, c),
#                    conv_block(c, c, c),
#                    conv_block(c, c, c)], block = DualInputSequential(block,[conv_block(c * 3, c, c),
#                    conv_block(c, c, c),
#                    conv_block(c, c, io_channels, is_last=True)]))
#
#        self.net = block
#
#        with torch.no_grad():
#            for param in self.net.parameters():
#                param *= 0.3
#
#    def forward(self, input, t, cond=None):
#
#        cA, cD = wavelet_decompose_batch(input)
#
#        timestep_embed = expand_to_planes(
#            self.timestep_embed(t[:, None]), cA.shape)
#
#        cA = torch.cat([cA, timestep_embed],dim=1) # not really sure about this
#        cD = torch.cat([cD, timestep_embed],dim=1)
#
#        return self.net(cA,cD)