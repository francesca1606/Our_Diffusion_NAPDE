import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary
from diffusion.blocks import SkipBlock, FourierFeatures, SelfAttention1d, ResConvBlock, Downsample1d, Upsample1d
from diffusion.utils import append_dims, expand_to_planes
from diffusion.utils import wavelet_decompose_batch, butterworth_decompose_batch


# MODELLO CHECKPOINT HUGO
class DiffusionAttnUnet1D(nn.Module):
    def __init__(
        self,
        io_channels=3,
        latent_dim=0,
        depth=4,
        n_attn_layers=6,
        # c_mults = [64,64,128,128] + [256]*10
        c_mults=[128, 128, 256, 256] + [512] * 10

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
        timestep_embed = expand_to_planes(
            self.timestep_embed(t[:, None]), input.shape)
        inputs = [input, timestep_embed]
        return self.net(torch.cat(inputs, dim=1))


class DiffusionAttnUnet1DCond(nn.Module):
    def __init__(
        self,
        io_channels=3,  # if you plan on using conditional input, set this to 6
        latent_dim=0,
        depth=4,
        n_attn_layers=6,
        c_mults=[128, 128, 256, 256] + [512] * 10

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
        self.lastconv = nn.Conv1d(io_channels, 3, 3, padding="same")

        with torch.no_grad():
            for param in self.net.parameters():
                param *= 0.5

    def forward(self, input, t, cond=None):
        timestep_embed = expand_to_planes(
            self.timestep_embed(t[:, None]), input.shape)
        inputs = [input, timestep_embed]
        if cond is not None:
            # cond = F.interpolate(cond, (input.shape[2], ), mode='linear', align_corners=False)
            inputs.append(cond)

        out = self.net(torch.cat(inputs, dim=1))
        out = self.lastconv(out)
        return out
