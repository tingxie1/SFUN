

import warnings
from collections import defaultdict

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

import numbers

from .base import BaseModel
from csi.data import gen_meas_torch_batch


try:
    from mamba_ssm.modules.mamba_simple import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: Mamba not available, using LargeKernelBlock instead")



def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b]", stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


def _align_to(x, ref):
    if x.shape[-2:] != ref.shape[-2:]:
        x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    return x


def mean_channels(F):
    assert F.dim() == 4
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert F.dim() == 4
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)



class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


CustomLayerNorm = LayerNorm


class SpectralAttention(nn.Module):

    def __init__(self, num_channels, kernel_size=5):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x):
        B, C, H, W = x.shape
        y = self.avg_pool(x)  
        y = y.view(B, 1, C)    
        y = self.conv(y)       
        y = self.sigmoid(y)
        y = y.view(B, C, 1, 1)
        return x * (self.gamma * y + (1 - self.gamma))


class SpectralSelfAttention(nn.Module):

    def __init__(self, num_channels, reduction=4):
        super().__init__()
        hidden = max(4, num_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.q = nn.Linear(num_channels, hidden)
        self.k = nn.Linear(num_channels, hidden)
        self.v = nn.Linear(num_channels, num_channels)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = hidden ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape

        y = self.avg_pool(x).view(B, C)

        q = self.q(y)  
        k = self.k(y)  
        v = self.v(y)  


        attn = torch.sigmoid(q * k * self.scale)  
        attn = attn.mean(dim=1, keepdim=True)     

        out = v * attn  
        out = out.view(B, C, 1, 1)

        return x + self.gamma * x * out



class LargeKernelBlock(nn.Module):

    def __init__(self, dim, kernel_size=21, expand_ratio=4):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2, groups=dim)
        self.norm = nn.GroupNorm(1, dim)
        self.pwconv1 = nn.Conv2d(dim, dim * expand_ratio, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim * expand_ratio, dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, guide=None):

        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return shortcut + self.gamma * x


class CrossLargeKernelBlock(nn.Module):

    def __init__(self, dim, kernel_size=21, expand_ratio=4):
        super().__init__()
        self.norm_x = nn.GroupNorm(1, dim)
        self.norm_g = nn.GroupNorm(1, dim)

        self.gate = nn.Conv2d(dim, dim, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)

        self.dwconv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2, groups=dim)
        self.norm = nn.GroupNorm(1, dim)
        self.pwconv1 = nn.Conv2d(dim, dim * expand_ratio, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim * expand_ratio, dim, 1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, guide):
        shortcut = x

        xn = self.norm_x(x)
        gn = self.norm_g(guide)
        x = xn + torch.sigmoid(self.gate(gn)) * self.proj(gn)

        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        return shortcut + self.gamma * x



if MAMBA_AVAILABLE:
    class SafeMamba(Mamba):
        def __init__(self, *args, **kwargs):
            kwargs["use_fast_path"] = False
            super().__init__(*args, **kwargs)

    class RCMambaBlock2D(nn.Module):
        def __init__(self, dim, d_state=16):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.mamba_row = SafeMamba(d_model=dim, d_state=d_state, expand=1)
            self.mamba_col = SafeMamba(d_model=dim, d_state=d_state, expand=1)
            self.alpha = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            B, C, H, W = x.shape
            xr = rearrange(x, "b c h w -> (b h) w c")
            xr = self.norm(xr)
            xr = self.mamba_row(xr)
            xr = rearrange(xr, "(b h) w c -> b c h w", b=B, h=H)

            xc = rearrange(x, "b c h w -> (b w) h c")
            xc = self.norm(xc)
            xc = self.mamba_col(xc)
            xc = rearrange(xc, "(b w) h c -> b c h w", b=B, w=W)

            return x + self.alpha * (xr + xc)

    class CrossRCMambaBlock2D(nn.Module):
        def __init__(self, dim, d_state=16, gate_r=8):
            super().__init__()
            self.norm0 = nn.GroupNorm(1, dim)
            self.norm1 = nn.GroupNorm(1, dim)
            self.cond_proj = nn.Conv2d(dim, dim, 1, bias=False)
            self.gate = nn.Conv2d(dim, dim, 1, bias=False)
            self.rc = RCMambaBlock2D(dim, d_state=d_state)
            self.alpha = nn.Parameter(torch.zeros(1))

            hidden = max(4, dim // gate_r)
            self.delta_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, hidden, 1, bias=True),
                nn.GELU(),
                nn.Conv2d(hidden, dim, 1, bias=True),
                nn.Sigmoid(),
            )

        def forward(self, x0, x1):
            x0n = self.norm0(x0)
            x1n = self.norm1(x1)
            cond = x0n + torch.sigmoid(self.gate(x1n)) * self.cond_proj(x1n)
            y = self.rc(cond)
            delta = y - cond
            delta = delta * self.delta_gate(x1n)
            return x0 + self.alpha * delta
else:

    RCMambaBlock2D = LargeKernelBlock
    CrossRCMambaBlock2D = CrossLargeKernelBlock



class Gated_Dconv_FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=True)
        self.act_fn = nn.GELU()
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = self.act_fn(x1) * x2
        x = self.project_out(x)
        return x



class EnhancedChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_feat = self.avg_pool(x)
        std_feat = stdv_channels(x)
        combined = torch.cat([avg_feat, std_feat], dim=1)
        attn = self.fc(combined)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.sigmoid(self.conv(y))
        return x * y



class GuideAlign(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.norm = nn.GroupNorm(1, out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.proj(x)))


class SimpleGateFuse(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.nx = nn.GroupNorm(1, ch)
        self.ng = nn.GroupNorm(1, ch)
        self.gate = nn.Conv2d(ch, ch, 1, bias=False)
        self.proj = nn.Conv2d(ch, ch, 1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, guide):
        x0 = self.nx(x)
        g0 = self.ng(guide)
        m = torch.sigmoid(self.gate(g0))
        delta = self.proj(g0) * m
        return x + self.alpha * delta


class SpatialSpectralSkipGate(nn.Module):

    def __init__(self, ch):
        super().__init__()
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 3, 1, 1, bias=False),
            nn.GroupNorm(1, ch),
            nn.Sigmoid()
        )
        self.spectral_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch * 2, ch // 4, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(ch // 4, ch, 1, bias=True),
            nn.Sigmoid()
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, skip, cond):
        cat_feat = torch.cat([skip, cond], dim=1)

        spatial_mask = self.spatial_gate(cat_feat)    
        spectral_mask = self.spectral_gate(cat_feat)  

        combined_mask = self.alpha * spatial_mask + (1 - self.alpha) * spectral_mask
        return skip * combined_mask



class SafeFFTFusion(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.pre_x = nn.Conv2d(channels, channels, 1, bias=False)
        self.pre_g = nn.Conv2d(channels, channels, 1, bias=False)

        self.amp_fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False)
        )

        self.gamma = nn.Parameter(torch.zeros(1))
        self.eps = 1e-8

    def forward(self, x, guide):
        B, C, H, W = x.shape


        xF = torch.fft.rfft2(self.pre_x(x), norm='backward')
        gF = torch.fft.rfft2(self.pre_g(guide), norm='backward')


        amp_x = torch.abs(xF)
        amp_g = torch.abs(gF)


        amp_fused = self.amp_fuse(torch.cat([amp_x, amp_g], dim=1))


        phase_x = xF / (amp_x + self.eps)

        out_F = phase_x * amp_fused
        out = torch.fft.irfft2(out_F, s=(H, W), norm='backward')

        return x + self.gamma * (out - x)


class SpaFreFusionV3(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.spa_process = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, 1, 1, bias=False),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
        )

        self.fre_process = SafeFFTFusion(channels)


        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.GroupNorm(1, channels),
            nn.GELU(),
        )


        self.spec_att = SpectralAttention(channels)

        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, guide):

        spa_out = self.spa_process(torch.cat([x, guide], dim=1))


        fre_out = self.fre_process(x, guide)


        fused = self.fuse(torch.cat([spa_out, fre_out], dim=1))


        fused = self.spec_att(fused)

        return x + self.alpha * fused



class ConvBlockGDFN(nn.Module):
    def __init__(self, in_ch, out_ch, ln_type="WithBias", ffn_expand=2.66,
                 res_scale=0.1, use_spectral_att=False):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

        self.norm1 = CustomLayerNorm(out_ch, ln_type)
        self.conv = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.act = nn.GELU()

        self.norm2 = CustomLayerNorm(out_ch, ln_type)
        self.ffn = Gated_Dconv_FeedForward(out_ch, ffn_expansion_factor=ffn_expand)

        self.ca = EnhancedChannelAttention(out_ch, reduction=8)

        self.use_spectral_att = use_spectral_att
        if use_spectral_att:
            self.spec_att = SpectralAttention(out_ch)

        self.res_scale = nn.Parameter(torch.tensor(res_scale))

    def forward(self, x):
        x = self.proj(x)
        x = x + self.res_scale * self.act(self.norm1(self.conv(x)))
        x = x + self.res_scale * self.ffn(self.norm2(x))
        x = self.ca(x)
        if self.use_spectral_att:
            x = self.spec_att(x)
        return x


class DownSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super().__init__()
        self.op = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(in_channels * 4, in_channels * 2, 1, bias=bias),
        )

    def forward(self, x):
        return self.op(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super().__init__()
        out_channels = in_channels // 2
        self.op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, 1, bias=bias),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.op(x)


class UNetCrossMambaFusion(nn.Module):

    def __init__(
        self,
        channels,
        ln_type="WithBias",
        use_pan=True,
        d_state=16,
        ffn_expand_hi=2.0,
        ffn_expand_lo=2.66,
        dec2_refine=True,
        use_skip_gate=True,
        use_dec_guide=True,
        use_mamba=True,
        use_spectral_att=True,
    ):
        super().__init__()
        C = channels
        self.use_pan = use_pan
        self.use_skip_gate = use_skip_gate
        self.use_dec_guide = use_dec_guide

        self.alpha = nn.Parameter(torch.tensor(0.1))


        self.ms_pre = nn.Conv2d(C, C, 3, 1, 1, bias=False)
        self.pan_pre = nn.Conv2d(C, C, 3, 1, 1, bias=False)


        self.g0 = GuideAlign(C, C)
        self.g1 = GuideAlign(C, 2 * C)
        self.g2 = GuideAlign(C, 4 * C)
        self.g3 = GuideAlign(C, 8 * C)


        self.enc0 = ConvBlockGDFN(C, C, ln_type=ln_type, ffn_expand=ffn_expand_hi,
                                   use_spectral_att=use_spectral_att)
        self.down1 = DownSample(C)
        self.enc1 = ConvBlockGDFN(2 * C, 2 * C, ln_type=ln_type, ffn_expand=ffn_expand_hi,
                                   use_spectral_att=use_spectral_att)

        self.down2 = DownSample(2 * C)
        self.enc2 = ConvBlockGDFN(4 * C, 4 * C, ln_type=ln_type, ffn_expand=ffn_expand_lo,
                                   use_spectral_att=False)

        self.down3 = DownSample(4 * C)
        self.enc3 = ConvBlockGDFN(8 * C, 8 * C, ln_type=ln_type, ffn_expand=ffn_expand_lo,
                                   use_spectral_att=False)


        self.fuse0 = SimpleGateFuse(C)
        self.fuse1 = SimpleGateFuse(2 * C)


        if use_mamba and MAMBA_AVAILABLE:
            self.bottleneck_block = CrossRCMambaBlock2D(8 * C, d_state=d_state)
        else:
            self.bottleneck_block = CrossLargeKernelBlock(8 * C, kernel_size=21)


        self.bottleneck_fuse = SpaFreFusionV3(8 * C)


        if use_skip_gate:
            self.skip_gate2 = SpatialSpectralSkipGate(4 * C)
            self.skip_gate1 = SpatialSpectralSkipGate(2 * C)
            self.skip_gate0 = SpatialSpectralSkipGate(C)


        self.up3 = UpSample(8 * C)
        self.dec2_main = ConvBlockGDFN(8 * C, 4 * C, ln_type=ln_type, ffn_expand=ffn_expand_lo)
        self.dec2_refine = ConvBlockGDFN(4 * C, 4 * C, ln_type=ln_type, ffn_expand=ffn_expand_lo) if dec2_refine else nn.Identity()
        self.dec2_guide = SimpleGateFuse(4 * C)

        self.up2 = UpSample(4 * C)
        self.dec1 = ConvBlockGDFN(4 * C, 2 * C, ln_type=ln_type, ffn_expand=ffn_expand_hi,
                                   use_spectral_att=use_spectral_att)

        self.up1 = UpSample(2 * C)
        self.dec0 = ConvBlockGDFN(2 * C, C, ln_type=ln_type, ffn_expand=ffn_expand_hi,
                                   use_spectral_att=use_spectral_att)

        if use_dec_guide:
            self.dec1_guide = SimpleGateFuse(2 * C)
            self.dec0_guide = SimpleGateFuse(C)

        self.out_proj = nn.Conv2d(C, C, 3, 1, 1, bias=False)

    def _build_guide(self, ms, pan):
        return pan if self.use_pan else ms

    def _down(self, x, s):
        return F.avg_pool2d(x, kernel_size=s, stride=s)

    def forward(self, msf, panf, **kwargs):
        panf = _align_to(panf, msf)
        msf = self.ms_pre(msf)
        panf = self.pan_pre(panf)

        guide = self._build_guide(msf, panf)

        g0 = self.g0(guide)
        g1 = self.g1(self._down(guide, 2))
        g2 = self.g2(self._down(guide, 4))
        g3 = self.g3(self._down(guide, 8))

        e0 = self.enc0(msf)
        e0 = self.fuse0(e0, g0)

        e1 = self.down1(e0)
        e1 = self.enc1(e1)
        e1 = self.fuse1(e1, g1)

        e2 = self.down2(e1)
        e2 = self.enc2(e2)

        e3 = self.down3(e2)
        e3 = self.enc3(e3)

        e3 = self.bottleneck_block(e3, g3)
        e3 = self.bottleneck_fuse(e3, g3)

        up3_out = self.up3(e3)

        if self.use_skip_gate:
            e2_gated = self.skip_gate2(e2, up3_out)
        else:
            e2_gated = e2

        d2 = torch.cat([up3_out, e2_gated], dim=1)
        d2 = self.dec2_main(d2)
        d2 = self.dec2_refine(d2)
        d2 = self.dec2_guide(d2, g2)

        up2_out = self.up2(d2)

        if self.use_skip_gate:
            e1_gated = self.skip_gate1(e1, up2_out)
        else:
            e1_gated = e1

        d1 = self.dec1(torch.cat([up2_out, e1_gated], dim=1))

        if self.use_dec_guide:
            d1 = self.dec1_guide(d1, g1)

        up1_out = self.up1(d1)

        if self.use_skip_gate:
            e0_gated = self.skip_gate0(e0, up1_out)
        else:
            e0_gated = e0

        d0 = self.dec0(torch.cat([up1_out, e0_gated], dim=1))

        if self.use_dec_guide:
            d0 = self.dec0_guide(d0, g0)

        return msf + self.alpha * self.out_proj(d0)



class XST(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        dim = cfg.MODEL.SFUN.DIM
        in_dim = cfg.MODEL.SFUN.IN_DIM
        out_dim = cfg.MODEL.SFUN.OUT_DIM
        ln_type = cfg.MODEL.SFUN.LAYERNORM_TYPE

        self.out_scale = nn.Parameter(torch.tensor(0.5))

        self.embeddingC = nn.Conv2d(in_dim, dim, 3, 1, 1, bias=False)

        self.pan_compress = nn.Conv2d(in_dim, 1, 1, 1, 0, bias=False)

        self.pan_edge_scale = nn.Parameter(torch.tensor(0.1))
        self.pan_edge = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        with torch.no_grad():
            k = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)
            self.pan_edge.weight.copy_(k.view(1, 1, 3, 3))
        self.pan_edge.weight.requires_grad_(False)

        self.embeddingP = nn.Conv2d(1, dim, 3, 1, 1, bias=False)

        self.spa_fre_blocks = nn.ModuleList([
            UNetCrossMambaFusion(
                dim,
                ln_type=ln_type,
                use_pan=True,
                d_state=16,
                ffn_expand_hi=2.0,
                ffn_expand_lo=2.66,
                dec2_refine=True,
                use_skip_gate=True,
                use_dec_guide=True,
                use_mamba=MAMBA_AVAILABLE,  
                use_spectral_att=True,       
            )
            for _ in range(1)
        ])

        self.mappingC = nn.Conv2d(dim, out_dim, 3, 1, 1, bias=False)

    def forward(self, inputs):
        xc = inputs["xc"]
        xp = inputs["xp"]

        xc0 = xc

        h_inp, w_inp = xc.shape[-2:]
        hb, wb = 16, 16
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        xc = F.pad(xc, [0, pad_w, 0, pad_h], mode="reflect")
        xp = F.pad(xp, [0, pad_w, 0, pad_h], mode="reflect")

        feat_c = self.embeddingC(xc)
        pan1 = self.pan_compress(xp)
        pan1 = pan1 + self.pan_edge_scale * self.pan_edge(pan1)
        feat_p = self.embeddingP(pan1)

        fused = feat_c
        for blk in self.spa_fre_blocks:
            fused = blk(fused, feat_p)

        out = self.mappingC(fused) + xc
        out = out[..., :h_inp, :w_inp]
        return xc0 + self.out_scale * (out - xc0)



def A(x, Phi):
    temp = x * Phi
    y = torch.sum(temp, 1)
    return y


def At(y, Phi):
    temp = torch.unsqueeze(y, 1).repeat(1, Phi.shape[1], 1, 1)
    x = temp * Phi
    return x


def shift_3d(inputs, step=2):
    B, C, H, W = inputs.shape
    out = inputs.new_zeros((B, C, H, W + (C - 1) * step))
    out[..., :W] = inputs
    for i in range(C):
        out[:, i:i + 1] = torch.roll(out[:, i:i + 1], shifts=step * i, dims=3)
    return out


def shift_back_3d(inputs, step=2):
    B, C, H, W = inputs.shape
    out = inputs.clone()
    for i in range(C):
        out[:, i:i + 1] = torch.roll(out[:, i:i + 1], shifts=-step * i, dims=3)
    return out


class DP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        def inv_softplus(x):
            return torch.log(torch.expm1(torch.tensor(x, dtype=torch.float32)))

        self.log_mu_c = nn.Parameter(inv_softplus(0.009))
        self.log_mu = nn.Parameter(inv_softplus(0.001))
        if self.cfg.DATASETS.WITH_PAN:
            self.log_mu_p = nn.Parameter(inv_softplus(0.009))

        self.eps = 1e-6

    def forward(self, inputs):
        out = defaultdict()
        z = inputs['z']
        yc = inputs['yc']
        phi_c = inputs['phi_c']

        mu_c = F.softplus(self.log_mu_c) + self.eps
        mu = F.softplus(self.log_mu) + self.eps
        if self.cfg.DATASETS.WITH_PAN:
            mu_p = F.softplus(self.log_mu_p) + self.eps
            xc_k_1 = inputs['xc']
            xp_k_1 = inputs['xp']

        B, C, HC, WC = z.shape
        phi_c_s = torch.sum(phi_c ** 2, 1)
        phi_c_s[phi_c_s == 0] = 1

        Phi_zc = A(z, phi_c)

        if self.cfg.DATASETS.WITH_PAN:
            Phi_xp_k_1 = A(xp_k_1, phi_c)
            denom = (mu + mu_c) + phi_c_s
            xc = (mu_c / (mu + mu_c)) * (z + At((yc - Phi_zc) / denom, phi_c)) \
                 + (mu / (mu + mu_c)) * (xp_k_1 + At((yc - Phi_xp_k_1) / denom, phi_c))
        else:
            denom = mu_c + phi_c_s
            xc = z + At((yc - Phi_zc) / denom, phi_c)

        out['xc'] = xc

        if self.cfg.DATASETS.WITH_PAN:
            yp = inputs['yp']
            phi_p = inputs['phi_p']

            B, C, HP, WP = phi_p.shape
            phi_p_s = torch.sum(phi_p ** 2, 1)
            phi_p_s[phi_p_s == 0] = 1

            z_img = shift_back_3d(z, step=self.cfg.DATASETS.STEP)[:, :, :, :WP]
            Phi_zp = A(z_img, phi_p)
            Phi_xc_k_1 = A(xc_k_1, phi_p)

            denom = (mu + mu_p) + phi_p_s
            xp = (mu_p / (mu + mu_p)) * (z_img + At((yp - Phi_zp) / denom, phi_p)) \
                 + (mu / (mu + mu_p)) * (xc_k_1 + At((yp - Phi_xc_k_1) / denom, phi_p))

            out['xp'] = xp

        return out



class sfun(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.fusion = nn.Conv2d(cfg.DATASETS.WAVE_LENS * 2, cfg.DATASETS.WAVE_LENS, 1, padding=0, bias=True)

        self.DP = nn.ModuleList([
            DP(cfg) for _ in range(cfg.MODEL.SFUN.STAGES)
        ]) if not cfg.MODEL.SFUN.SHARE_PARAMS else DP(cfg)

        self.PP = nn.ModuleList([
            XST(cfg) for _ in range(cfg.MODEL.SFUN.STAGES)
        ]) if not cfg.MODEL.SFUN.SHARE_PARAMS else XST(cfg)

        self.spec_head = SpectralMLPRefine(
            ch=cfg.DATASETS.WAVE_LENS,
            expand=2,
            res_scale=0.2,
            use_spatial=True,
            use_spectral=True  
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def prepare_input(self, data):
        hsi = data['hsi']
        mask = data['mask']

        Meas = gen_meas_torch_batch(hsi, mask, step=self.cfg.DATASETS.STEP,
                                     wave_len=self.cfg.DATASETS.WAVE_LENS,
                                     mask_type=self.cfg.DATASETS.MASK_TYPE,
                                     with_noise=self.cfg.DATASETS.TRAIN.WITH_NOISE,
                                     with_pan=self.cfg.DATASETS.WITH_PAN,
                                     direction=self.cfg.DATASETS.DISPERSION_DIRECTION)

        data['MeasC'] = Meas['MeasC']
        data['MeasH'] = Meas['MeasH']
        data['PhiC'] = mask

        if self.cfg.DATASETS.WITH_PAN:
            data['MeasP'] = Meas['MeasP']
            data['PhiP'] = torch.ones_like(data['MeasH'])

        return data

    def initial(self, y, Phi):
        nC = self.cfg.DATASETS.WAVE_LENS
        step = self.cfg.DATASETS.STEP
        bs, nC, row, col = Phi.shape
        y_shift = torch.zeros(bs, nC, row, col).to(y.device).float()
        for i in range(nC):
            y_shift[:, i, :, step * i:step * i + col - (nC - 1) * step] = y[:, :, step * i:step * i + col - (nC - 1) * step]
        z = self.fusion(torch.cat([y_shift, Phi], dim=1))
        return z

    def forward_train(self, data):
        yc = data['MeasC']
        x = data['MeasH']
        phi_c = data['PhiC']

        if self.cfg.DATASETS.WITH_PAN:
            phi_p = data['PhiP']
            yp = data['MeasP']

        B, nC, H_, W_ = x.shape

        if self.cfg.EXPERIMENTAL_TYPE == "real":
            yc = yc * nC / 2 / 1.2
            if self.cfg.DATASETS.WITH_PAN:
                yp = yp * nC / 1.2

        z = self.initial(yc, phi_c)

        xc_k_1 = shift_back_3d(z, step=self.cfg.DATASETS.STEP)[:, :, :, :W_]
        xp_k_1 = z

        for i in range(self.cfg.MODEL.SFUN.STAGES):
            inputs = (
                {"yc": yc, "yp": yp, "z": z, "xc": xc_k_1, "xp": xp_k_1, "phi_c": phi_c, "phi_p": phi_p}
                if self.cfg.DATASETS.WITH_PAN
                else {"yc": yc, "z": z, "phi_c": phi_c}
            )

            out = self.DP[i](inputs) if not self.cfg.MODEL.SFUN.SHARE_PARAMS else self.DP(inputs)

            xc_k_1 = shift_back_3d(out["xc"], step=self.cfg.DATASETS.STEP)[:, :, :, :W_]
            xp_k_1 = shift_3d(out["xp"], step=self.cfg.DATASETS.STEP)

            pp_inputs = {"xc": xc_k_1, "xp": out["xp"]}

            if self.cfg.MODEL.SFUN.SHARE_PARAMS:
                z_img = self.PP(pp_inputs)
            else:
                z_img = self.PP[i](pp_inputs)

            if i == self.cfg.MODEL.SFUN.STAGES - 1:
                z_img = self.spec_head(z_img)

            z = shift_3d(z_img, step=self.cfg.DATASETS.STEP)
            z = z[..., :phi_c.shape[-1]]

        z_img_out = shift_back_3d(z, step=self.cfg.DATASETS.STEP)[:, :, :, :W_]
        return z_img_out

    def forward_test(self, data):
        return self.forward_train(data)



class sfun_Profiling(BaseModel):


    def __init__(self, cfg):
        if not MAMBA_AVAILABLE:
            raise ImportError("sfun_Profiling requires Mamba; convolution fallback is disabled.")
        if cfg.DATASETS.WAVE_LENS != 28 or not cfg.DATASETS.WITH_PAN:
            raise ValueError("sfun_Profiling expects 28 spectral bands with PAN guidance.")

        cfg = cfg.clone()
        cfg.defrost()
        cfg.MODEL.SFUN.STAGES = 9
        cfg.MODEL.SFUN.SHARE_PARAMS = True
        cfg.MODEL.SFUN.DIM = 16
        cfg.MODEL.SFUN.IN_DIM = 28
        cfg.MODEL.SFUN.OUT_DIM = 28
        cfg.freeze()
        super().__init__(cfg)

        self.fusion = nn.Conv2d(cfg.DATASETS.WAVE_LENS * 2, cfg.DATASETS.WAVE_LENS, 1, padding=0, bias=True)

        self.DP = nn.ModuleList([
            DP(cfg) for _ in range(cfg.MODEL.SFUN.STAGES)
        ]) if not cfg.MODEL.SFUN.SHARE_PARAMS else DP(cfg)

        self.PP = nn.ModuleList([
            XST(cfg) for _ in range(cfg.MODEL.SFUN.STAGES)
        ]) if not cfg.MODEL.SFUN.SHARE_PARAMS else XST(cfg)

        self.spec_head = SpectralMLPRefine(
            ch=cfg.DATASETS.WAVE_LENS,
            expand=2,
            res_scale=0.2,
            use_spatial=True,
            use_spectral=True
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def initial(self, y, Phi):
        nC = self.cfg.DATASETS.WAVE_LENS
        step = self.cfg.DATASETS.STEP
        bs, nC, row, col = Phi.shape
        y_shift = torch.zeros(bs, nC, row, col).to(y.device).float()
        for i in range(nC):
            y_shift[:, i, :, step * i:step * i + col - (nC - 1) * step] = y[:, :, step * i:step * i + col - (nC - 1) * step]
        z = self.fusion(torch.cat([y_shift, Phi], dim=1))
        return z

    def forward(self, yc, yp, phi_c, phi_p, x):
        B, nC, H_, W_ = x.shape

        if self.cfg.EXPERIMENTAL_TYPE == "real":
            yc = yc * nC / 2 / 1.2
            if self.cfg.DATASETS.WITH_PAN:
                yp = yp * nC / 1.2

        z = self.initial(yc, phi_c)
        xc_k_1, xp_k_1 = shift_back_3d(z, step=self.cfg.DATASETS.STEP)[:, :, :, :W_], z

        for i in range(self.cfg.MODEL.SFUN.STAGES):
            inputs = {"yc": yc, "yp": yp, "z": z, "xc": xc_k_1, "xp": xp_k_1, "phi_c": phi_c, "phi_p": phi_p} if self.cfg.DATASETS.WITH_PAN else {"yc": yc, "z": z, "phi_c": phi_c}
            out = self.DP[i](inputs) if not self.cfg.MODEL.SFUN.SHARE_PARAMS else self.DP(inputs)

            xc_k_1 = shift_back_3d(out["xc"], step=self.cfg.DATASETS.STEP)[:, :, :, :W_]
            xp_k_1 = shift_3d(out["xp"], step=self.cfg.DATASETS.STEP)

            pp_inputs = {"xc": xc_k_1, "xp": out["xp"]}

            if self.cfg.MODEL.SFUN.SHARE_PARAMS:
                z_img = self.PP(pp_inputs)
            else:
                z_img = self.PP[i](pp_inputs)

            if i == self.cfg.MODEL.SFUN.STAGES - 1:
                z_img = self.spec_head(z_img)

            z = shift_3d(z_img, step=self.cfg.DATASETS.STEP)
            z = z[..., :phi_c.shape[-1]]

        z = shift_back_3d(z, step=self.cfg.DATASETS.STEP)[:, :, :, :W_]
        return z
