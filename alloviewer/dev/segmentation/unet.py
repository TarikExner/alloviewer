import torch
import torch.nn as nn
import torch.nn.functional as F

class DWConvBlock(nn.Module):
    """
    Apply two depthwise/pointwise convolution pairs with normalization and ReLU.
    """
    def __init__(self, in_ch, out_ch, norm="group", groups=8, p_drop=0.0):
        """
        Build depthwise and pointwise layers, normalization, and optional dropout.

        Parameters
        ----------
        in_ch : int
            Number of input feature channels.
        out_ch : int
            Number of output feature channels.
        norm : str
            Normalization mode selecting group, batch, or identity normalization. Default is
            ``'group'``.
        groups : int
            Requested group count for group normalization. Default is ``8``.
        p_drop : float
            Probability of channel-wise dropout. Default is ``0.0``.
        """
        super().__init__()
        # depthwise 3x3 then pointwise 1x1 (two times)
        self.dw1 = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pw1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.dw2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False)
        self.pw2 = nn.Conv2d(out_ch, out_ch, 1, bias=False)

        if norm == "batch":
            self.n1 = nn.BatchNorm2d(out_ch)
            self.n2 = nn.BatchNorm2d(out_ch)
        elif norm == "group":
            g = min(groups, out_ch)
            self.n1 = nn.GroupNorm(g, out_ch)
            self.n2 = nn.GroupNorm(g, out_ch)
        else:
            self.n1 = nn.Identity()
            self.n2 = nn.Identity()

        self.drop = nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Apply the block to a batch of channel-first feature maps.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch, channels, height, width).

        Returns
        -------
        torch.Tensor
            Channel-first output feature maps; the full UNet returns logits without applying
            sigmoid.
        """
        x = self.dw1(x)
        x = self.pw1(x)
        x = self.n1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.dw2(x)
        x = self.pw2(x)
        x = self.n2(x)
        x = self.act(x)
        return x


class Down(nn.Module):
    """
    Downsample spatial dimensions with max pooling before a convolution block.
    """
    def __init__(self, in_ch, out_ch, **kw):
        """
        Create the pooling layer and depthwise convolution block.

        Parameters
        ----------
        in_ch : int
            Number of input feature channels.
        out_ch : int
            Number of output feature channels.
        **kw : dict
            Additional keyword arguments forwarded to ``DWConvBlock``.
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = DWConvBlock(in_ch, out_ch, **kw)

    def forward(self, x):
        """
        Pool the input by two and apply the convolution block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch, channels, height, width).

        Returns
        -------
        torch.Tensor
            Channel-first output feature maps; the full UNet returns logits without applying
            sigmoid.
        """
        return self.block(self.pool(x))


class Up(nn.Module):
    """
    Upsample decoder features and combine them with an encoder skip connection.
    """
    def __init__(self, in_ch, out_ch, bilinear=True, **kw):
        """
        Create bilinear upsampling and the convolution block for concatenated features.

        Parameters
        ----------
        in_ch : int
            Number of input feature channels.
        out_ch : int
            Number of output feature channels.
        bilinear : bool
            Accepted for compatibility; this implementation always uses bilinear upsampling.
            Default is ``True``.
        **kw : dict
            Normalization and dropout options forwarded to DWConvBlock.
        """
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.block = DWConvBlock(in_ch, out_ch, **kw)

    def forward(self, x, skip):
        """
        Upsample, align spatial dimensions, concatenate the skip map, and convolve.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch, channels, height, width).
        skip : torch.Tensor
            Encoder feature tensor concatenated along the channel axis after spatial alignment.

        Returns
        -------
        torch.Tensor
            Channel-first output feature maps; the full UNet returns logits without applying
            sigmoid.
        """
        x = self.up(x)
        # pad if shapes differ by 1 due to odd sizes
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        if diffY or diffX:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.block(x)


class UNetOptimized(nn.Module):
    """
    Small U-Net for 2D segmentation.
    Defaults: RGB input (3), two outputs (cell, boundary).
    Set out_channels=1 if you want a single mask.
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=4,
        base_ch=16,       # 16 is fast on CPU; 32 if you want more capacity
        norm="group",     # "group", "batch", or None
        groups=8,
        p_drop=0.0,
    ):
        """
        Build the encoder, bottleneck, decoder, and output layer and initialize weights.

        Parameters
        ----------
        in_channels : int
            Number of model input channels. Default is ``3``.
        out_channels : int
            Number of model output channels. Default is ``4``.
        base_ch : int
            Base number of feature channels in the UNet. Default is ``16``.
        norm : str
            Normalization mode selecting group, batch, or identity normalization. Default is
            ``'group'``.
        groups : int
            Requested group count for group normalization. Default is ``8``.
        p_drop : float
            Probability of channel-wise dropout. Default is ``0.0``.
        """
        super().__init__()
        c = base_ch
        kw = dict(norm=norm, groups=groups, p_drop=p_drop)

        # encoder
        self.inc   = DWConvBlock(in_channels, c, **kw)
        self.down1 = Down(c, c*2, **kw)
        self.down2 = Down(c*2, c*4, **kw)
        self.down3 = Down(c*4, c*8, **kw)   # depth 4 (stop at /16)
        # bottleneck kept narrow for speed
        self.bott  = Down(c*8, c*8, **kw)

        # decoder
        self.up1 = Up(c*16, c*4, **kw)
        self.up2 = Up(c*8, c*2, **kw)
        self.up3 = Up(c*4, c, **kw)
        self.up4 = Up(c*2, c, **kw)

        self.outc = nn.Conv2d(c, out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """
        Initialize convolution weights with Kaiming normal values and reset normalization layers.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d,)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Run the encoder and decoder and return per-pixel output logits.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch, channels, height, width).

        Returns
        -------
        torch.Tensor
            Channel-first output feature maps; the full UNet returns logits without applying
            sigmoid.
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        xb = self.bott(x4)

        x  = self.up1(xb, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.outc(x)


def build_unet_cpu_small(in_channels=3, out_channels=4) -> UNetOptimized:
    """
    Good default for CPU inference.

    Parameters
    ----------
    in_channels : int
        Number of model input channels. Default is ``3``.
    out_channels : int
        Number of model output channels. Default is ``4``.

    Returns
    -------
    UNetOptimized
        Initialized UNet with 16 base channels, group normalization, and no dropout. Model
        weights are newly initialized.
    """
    return UNetOptimized(in_channels=in_channels, out_channels=out_channels,
                         base_ch=16, norm="group", groups=8, p_drop=0.0)

def build_unet_cpu_medium(in_channels=3, out_channels=4) -> UNetOptimized:
    """
    A bit more capacity, still CPU-friendly.

    Parameters
    ----------
    in_channels : int
        Number of model input channels. Default is ``3``.
    out_channels : int
        Number of model output channels. Default is ``4``.

    Returns
    -------
    UNetOptimized
        Initialized UNet with 32 base channels, group normalization, and no dropout. Model
        weights are newly initialized.
    """
    return UNetOptimized(in_channels=in_channels, out_channels=out_channels,
                         base_ch=32, norm="group", groups=8, p_drop=0.0)

def build_unet_cpu_large(in_channels=3, out_channels=4) -> UNetOptimized:
    """
    A bit more capacity, still CPU-friendly.

    Parameters
    ----------
    in_channels : int
        Number of model input channels. Default is ``3``.
    out_channels : int
        Number of model output channels. Default is ``4``.

    Returns
    -------
    UNetOptimized
        Initialized UNet with 64 base channels, group normalization, and no dropout. Model
        weights are newly initialized.
    """
    return UNetOptimized(in_channels=in_channels, out_channels=out_channels,
                         base_ch=64, norm="group", groups=8, p_drop=0.0)

