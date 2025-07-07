import math
import torch
from torch import nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)

# Noise level (and other) conditioning
class ResConvBlock(ResidualBlock):
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
    def __init__(self, *main):
        super().__init__()
        self.main = nn.Sequential(*main)
    def forward(self, input):
        n = self.main(input).shape[2]
        return torch.cat([self.main(input), input], dim=1)

class FourierFeatures(nn.Module):
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





# -----------------------------------------
# NEW BLOCKS
# -----------------------------------------

class SequentialY(nn.Module):
    def __init__(self, main1_layers, main2_layers, block):
        super().__init__()
        self.main1 = nn.Sequential(*main1_layers)
        self.main2 = nn.Sequential(*main2_layers)
        self.core_block = block

    def forward(self, input1, input2):
        assert input1.shape == input2.shape

        out1 = self.main1(input1)
        out2 = self.main2(input2)

        out = self.core_block(out1, out2)

        # no skip connection
        return out


class DualInputSequential(nn.Module):

    def __init__(self, block, other_layers):
        super().__init__()
        self.block = block
        # Another nn.Sequential or module
        self.other_layers = nn.Sequential(*other_layers)

    def forward(self, x1, x2):
        x = self.block(x1, x2)  # supports dual input
        x = self.other_layers(x)  # apply to single output
        return x


class HalfChannels1d(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        out_channels = in_channels // 2
        self.channel_reducer = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.channel_reducer(x)


class CoreBlock(nn.Module):
    def __init__(self, kernel = "cubic", c_in = 512):
        super().__init__()
        self.halfchannels = HalfChannels1d(c_in)

    def forward(self, input1, input2):
        assert input1.shape == input2.shape
        out = torch.cat([input1, input2], dim=1)
        return self.halfchannels(out)
        


class CrossAttention1d(nn.Module):
    def __init__(self, c_in, n_head=1, dropout_rate=0.):
        super().__init__()
        assert c_in % n_head == 0
        self.n_head = n_head
        self.head_dim = c_in // n_head
        self.scale = self.head_dim ** -0.5 

        self.norm_a = nn.GroupNorm(1, c_in)
        self.norm_b = nn.GroupNorm(1, c_in)

        self.qkv_proj_a = nn.Conv1d(c_in, c_in * 3, 1)
        self.qkv_proj_b = nn.Conv1d(c_in, c_in * 3, 1)

        self.out_proj_a = nn.Conv1d(c_in, c_in, 1)
        self.out_proj_b = nn.Conv1d(c_in, c_in, 1)
        
        self.dropout = nn.Dropout(dropout_rate, inplace=True)

    def _reshape(self, x):
        # x: (n, c, l) → (n, n_head, l, head_dim)
        n, _, l = x.shape
        return x.view(n, self.n_head, self.head_dim, l).transpose(2, 3)

    def forward(self, a, b):

        assert a.shape == b.shape

        n, c, l = a.shape

        # Normalize
        a_norm = self.norm_a(a)
        b_norm = self.norm_b(b)

        # Queries keys and values
        q_a, k_a, v_a = self.qkv_proj_a(a_norm).chunk(3, dim=1)
        q_b, k_b, v_b = self.qkv_proj_b(b_norm).chunk(3, dim=1)

        # Reshape to devide each head
        q_a, k_b, v_b = map(self._reshape, (q_a, k_b, v_b))
        q_b, k_a, v_a = map(self._reshape, (q_b, k_a, v_a))

        # Cross attention: a attends to b
        attn_ab = (q_a @ k_b.transpose(-2, -1)) * self.scale
        attn_ab = F.softmax(attn_ab, dim=-1)
        out_a = (attn_ab @ v_b).transpose(2, 3).contiguous().view(n, c, l)

        # Cross attention: b attends to a
        attn_ba = (q_b @ k_a.transpose(-2, -1)) * self.scale
        attn_ba = F.softmax(attn_ba, dim=-1)
        out_b = (attn_ba @ v_a).transpose(2, 3).contiguous().view(n, c, l)

        # Apply output projection + residual
        out_a = self.dropout(self.out_proj_a(out_a)) + a
        out_b = self.dropout(self.out_proj_b(out_b)) + b

        return out_a, out_b
 
class CrossAttentionSkipBlockY(nn.Module):
    def __init__(self, main1_layers, main2_layers, block, channels, n_heads, dropout_rate=0.3):
        super().__init__()
        self.main1 = nn.Sequential(*main1_layers)
        self.main2 = nn.Sequential(*main2_layers)
        self.core_block = block
        self.CrossAttention = CrossAttention1d(c_in=channels,n_head=n_heads, dropout_rate=dropout_rate)

    def forward(self, input1, input2):
        assert input1.shape == input2.shape

        out1, out2 = self.CrossAttention(input1, input2)

        out1 = self.main1(out1)
        out2 = self.main2(out2)

        out = self.core_block(out1, out2)

        return torch.cat([out, input1, input2], dim=1)



# -------------------
# wavelet
# -------------------

class Upsample1dHalfChannels(nn.Module):
    def __init__(self, kernel='linear', pad_mode='reflect'):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor(_kernels[kernel]) * 2
        self.pad = kernel_1d.shape[0] // 2 - 1
        self.register_buffer('kernel', kernel_1d)
        # channel_reducer will be created dynamically in forward

    def forward(self, x):
        x = F.pad(x, ((self.pad + 1) // 2,) * 2, self.pad_mode)
        weight = x.new_zeros([x.shape[1], x.shape[1], self.kernel.shape[0]])
        indices = torch.arange(x.shape[1], device=x.device)
        weight[indices, indices] = self.kernel.to(weight)
        x = F.conv_transpose1d(x, weight, stride=2, padding=self.pad * 2 + 1)
        # Dynamically create channel reducer if not present or if channel count changes
        out_channels = x.shape[1] // 2
        if not hasattr(self, 'channel_reducer') or self.channel_reducer.in_channels != x.shape[1]:
            self.channel_reducer = nn.Conv1d(x.shape[1], out_channels, 1).to(x.device)
        x = self.channel_reducer(x)
        return x
        
#class CoreBlock(nn.Module):
#  def __init__(self, kernel = "cubic"):
#    super().__init__()
#    self.upsample = Upsample1dHalfChannels(kernel)
#
#  def forward(self, input1, input2):
#    assert input1.shape == input2.shape
#    out = torch.cat([input1, input2], dim=1)
#    return self.upsample(out)
#    
#class CrossAttention1d(nn.Module):
#    def __init__(self, c_in, n_head=1, dropout_rate=0.):
#        super().__init__()
#        assert c_in % n_head == 0
#        self.n_head = n_head
#        self.head_dim = c_in // n_head
#        self.scale = self.head_dim ** -0.5 #why use this as scale? (check it, shouldn't it be scale = k.shape[3]**-0.25?)
#
#        self.norm_a = nn.GroupNorm(1, c_in)
#        self.norm_b = nn.GroupNorm(1, c_in)
#
#        self.q_proj_a = nn.Conv1d(c_in, c_in, 1)
#        self.kv_proj_a = nn.Conv1d(c_in, c_in * 2, 1)
#
#        self.q_proj_b = nn.Conv1d(c_in, c_in, 1)
#        self.kv_proj_b = nn.Conv1d(c_in, c_in * 2, 1)
#
#        self.out_proj_a = nn.Conv1d(c_in, c_in, 1)
#        self.out_proj_b = nn.Conv1d(c_in, c_in, 1)
#        self.dropout = nn.Dropout(dropout_rate, inplace=True)
#
#    def _reshape(self, x):
#        # x: (n, c, l) → (n, n_head, l, head_dim)
#        n, _, l = x.shape
#        return x.view(n, self.n_head, self.head_dim, l).transpose(2, 3)
#
#    def forward(self, a, b):
#
#        assert a.shape == b.shape
#
#        n, c, l = a.shape
#
#        # Normalize
#        a_norm = self.norm_a(a)
#        b_norm = self.norm_b(b)
#
#        # Queries
#        q_a = self.q_proj_a(a_norm)
#        q_b = self.q_proj_b(b_norm)
#
#        # Keys, Values
#        k_a, v_a = self.kv_proj_a(a_norm).chunk(2, dim=1)
#        k_b, v_b = self.kv_proj_b(b_norm).chunk(2, dim=1)
#
#
#        # Reshape to devide each head
#        q_a, k_b, v_b = map(self._reshape, (q_a, k_b, v_b))
#        q_b, k_a, v_a = map(self._reshape, (q_b, k_a, v_a))
#
#        # Cross attention: a attends to b
#        attn_ab = (q_a @ k_b.transpose(-2, -1)) * self.scale
#        attn_ab = F.softmax(attn_ab, dim=-1)
#        out_a = (attn_ab @ v_b).transpose(2, 3).contiguous().view(n, c, l)
#
#        # Cross attention: b attends to a
#        attn_ba = (q_b @ k_a.transpose(-2, -1)) * self.scale
#        attn_ba = F.softmax(attn_ba, dim=-1)
#        out_b = (attn_ba @ v_a).transpose(2, 3).contiguous().view(n, c, l)
#
#        # Apply output projection + residual
#        out_a = self.dropout(self.out_proj_a(out_a)) + a
#        out_b = self.dropout(self.out_proj_b(out_b)) + b
#
#        return out_a, out_b
#        
#class CrossAttentionSkipBlockY(nn.Module):
#    def __init__(self, main1_layers, main2_layers, block, channels, n_heads, dropout_rate=0.3,kernel = "cubic"):
#        super().__init__()
#        self.main1 = nn.Sequential(*main1_layers)
#        self.main2 = nn.Sequential(*main2_layers)
#        self.core_block = block
#        self.CrossAttention = CrossAttention1d(c_in=channels,n_head=n_heads, dropout_rate=dropout_rate)
#        self.upsample = Upsample1d(kernel)
#
#    def forward(self, input1, input2):
#        assert input1.shape == input2.shape
#
#        out1, out2 = self.CrossAttention(input1, input2)
#
#        out1 = self.main1(out1)
#        out2 = self.main2(out2)
#
#        out = self.core_block(out1, out2)
#
#        return torch.cat([out, self.upsample(input1), self.upsample(input2)], dim=1)#