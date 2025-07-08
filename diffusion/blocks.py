import math
import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """
    Residual Block with skip connection.

    Args:
        main (list): List of layers to be applied in the main path.
        skip (nn.Module, optional): Skip connection module. Defaults to nn.Identity().
    """

    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)


class ResConvBlock(ResidualBlock):
    """
    Residual Convolutional Block. 

    Args:
        c_in (int): Number of input channels.
        c_mid (int): Number of intermediate channels.
        c_out (int): Number of output channels.
        is_last (bool, optional): If True, applies no normalization or activation at the end. Defaults to False.
    """

    def __init__(self, c_in, c_mid, c_out, is_last=False):
        skip = None if c_in == c_out else nn.Conv1d(c_in, c_out, 1, bias=False)
        super().__init__([
            nn.Conv1d(c_in, c_mid, 5, padding=2),
            nn.GroupNorm(1, c_mid),
            nn.GELU(),
            nn.Conv1d(c_mid, c_out, 5, padding=2),
            nn.GroupNorm(1, c_out) if not is_last else nn.Identity(),
            nn.GELU() if not is_last else nn.Identity(),
        ], skip)


class SelfAttention1d(nn.Module):
    """
    Self-Attention Block for 1D inputs.

    Args:
        c_in (int): Number of input channels.
        n_head (int, optional): Number of attention heads. Defaults to 1.
        dropout_rate (float, optional): Dropout rate for the output. Defaults to 0.
    """

    def __init__(self, c_in, n_head=1, dropout_rate=0.):
        super().__init__()
        assert c_in % n_head == 0
        self.norm = nn.GroupNorm(1, c_in)
        self.n_head = n_head
        self.qkv_proj = nn.Conv1d(c_in, c_in * 3, 1)
        self.out_proj = nn.Conv1d(c_in, c_in, 1)
        self.dropout = nn.Dropout(dropout_rate, inplace=True)

    def forward(self, input):
        n, c, s = input.shape
        qkv = self.qkv_proj(self.norm(input))
        qkv = qkv.view(
            [n, self.n_head * 3, c // self.n_head, s]).transpose(2, 3)
        q, k, v = qkv.chunk(3, dim=1)
        scale = k.shape[3]**-0.25
        att = ((q * scale) @ (k.transpose(2, 3) * scale)).softmax(3)
        y = (att @ v).transpose(2, 3).contiguous().view([n, c, s])
        return input + self.dropout(self.out_proj(y))


class SkipBlock(nn.Module):
    """
    Skip Block that concatenates the input with the output of the main path.

    Args:
        main (list): List of layers to be applied in the main path.
    """

    def __init__(self, *main):
        super().__init__()
        self.main = nn.Sequential(*main)

    def forward(self, input):
        return torch.cat([self.main(input), input], dim=1)


class FourierFeatures(nn.Module):
    """
    Fourier Features for positional encoding.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features, must be even.
        std (float, optional): Standard deviation for the weight initialization. Defaults to 1.
    """

    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn(
            [out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


_kernels = {
    'linear':
        [1 / 8, 3 / 8, 3 / 8, 1 / 8],
    'cubic':
        [-0.01171875, -0.03515625, 0.11328125, 0.43359375,
         0.43359375, 0.11328125, -0.03515625, -0.01171875],
    'lanczos3':
        [0.003689131001010537, 0.015056144446134567, -0.03399861603975296,
         -0.066637322306633, 0.13550527393817902, 0.44638532400131226,
         0.44638532400131226, 0.13550527393817902, -0.066637322306633,
         -0.03399861603975296, 0.015056144446134567, 0.003689131001010537]
}


class Downsample1d(nn.Module):
    """
    Downsample 1D input using a specified kernel. Consistently halves the input length.
    If the input length is odd it will be rounded down.

    Args:
        kernel (str): Type of kernel to use for downsampling. Options are 'linear', 'cubic', 'lanczos3'.
        pad_mode (str): Padding mode to use. Defaults to 'reflect'.
    """

    def __init__(self, kernel='linear', pad_mode='reflect'):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor(_kernels[kernel])
        self.pad = kernel_1d.shape[0] // 2 - 1
        self.register_buffer('kernel', kernel_1d)

    def forward(self, x):
        x = F.pad(x, (self.pad,) * 2, self.pad_mode)
        weight = x.new_zeros([x.shape[1], x.shape[1], self.kernel.shape[0]])
        indices = torch.arange(x.shape[1], device=x.device)
        weight[indices, indices] = self.kernel.to(weight)
        return F.conv1d(x, weight, stride=2)


class Upsample1d(nn.Module):
    """
    Upsample 1D input using a specified kernel. Consistently doubles the input length.

    Args:
        kernel (str): Type of kernel to use for upsampling. Options are 'linear', 'cubic', 'lanczos3'.
        pad_mode (str): Padding mode to use. Defaults to 'reflect'.
    """

    def __init__(self, kernel='linear', pad_mode='reflect'):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor(_kernels[kernel]) * 2
        self.pad = kernel_1d.shape[0] // 2 - 1
        self.register_buffer('kernel', kernel_1d)

    def forward(self, x):
        x = F.pad(x, ((self.pad + 1) // 2,) * 2, self.pad_mode)
        weight = x.new_zeros([x.shape[1], x.shape[1], self.kernel.shape[0]])
        indices = torch.arange(x.shape[1], device=x.device)
        weight[indices, indices] = self.kernel.to(weight)
        return F.conv_transpose1d(x, weight, stride=2, padding=self.pad * 2 + 1)
