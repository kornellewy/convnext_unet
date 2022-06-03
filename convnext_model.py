"""
source:
https://arxiv.org/pdf/2201.03545.pdf
https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from zmq import device


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNectBlock(nn.Module):
    r"""ConvNeXt ConvNectBlock. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x


class UpConvNext(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.upscale_factor = 2
        self.up = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(
            ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.norm = LayerNorm(ch_out, eps=1e-6)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.act(x)
        return x


class UpConvNext2(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.upscale_factor = 2
        self.pixel = nn.PixelShuffle(upscale_factor=self.upscale_factor)
        self.up = nn.ConvTranspose2d(
            ch_in // self.upscale_factor ** 2,
            ch_out,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.norm = LayerNorm(ch_out, eps=1e-6)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.pixel(x)
        x = self.up(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.act(x)
        return x


class ConvNext_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv_next = nn.Sequential(ConvNectBlock(dim=ch_out))
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        return self.conv_next(x)


class U_ConvNext(nn.Module):
    """
    Version 1 using jast ConvNext_block. No bach normalization and no relu.
    """

    def __init__(self, img_ch=3, output_ch=1, channels=24):
        super().__init__()
        self.Maxpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

        self.Conv1 = ConvNext_block(ch_in=img_ch, ch_out=channels)
        self.Conv2 = ConvNext_block(ch_in=channels, ch_out=channels * 2)
        self.Conv3 = ConvNext_block(ch_in=channels * 2, ch_out=channels * 4)
        self.Conv4 = ConvNext_block(ch_in=channels * 4, ch_out=channels * 8)
        self.Conv5 = ConvNext_block(ch_in=channels * 8, ch_out=channels * 16)

        self.Up5 = UpConvNext2(ch_in=channels * 16, ch_out=channels * 8)
        self.Up_conv5 = ConvNext_block(ch_in=channels * 16, ch_out=channels * 8)

        self.Up4 = UpConvNext2(ch_in=channels * 8, ch_out=channels * 4)
        self.Up_conv4 = ConvNext_block(ch_in=channels * 8, ch_out=channels * 4)

        self.Up3 = UpConvNext2(ch_in=channels * 4, ch_out=channels * 2)
        self.Up_conv3 = ConvNext_block(ch_in=channels * 4, ch_out=channels * 2)

        self.Up2 = UpConvNext2(ch_in=channels * 2, ch_out=channels)
        self.Up_conv2 = ConvNext_block(ch_in=channels * 2, ch_out=channels)

        self.Conv_1x1 = nn.Conv2d(
            channels, output_ch, kernel_size=1, stride=1, padding=0
        )
        self.last_activation = nn.Hardtanh()

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x2 = self.dropout(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x3 = self.dropout(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x4 = self.dropout(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        x5 = self.dropout(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = self.last_activation(d1)

        return d1


class ConvNextDiscriminator(nn.Module):
    def __init__(self, in_channels=3, channels=4):
        super().__init__()

        self.Maxpool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.block1 = ConvNext_block(ch_in=in_channels, ch_out=channels * 2)
        self.block2 = ConvNext_block(ch_in=channels * 2, ch_out=channels * 4)
        self.block3 = ConvNext_block(ch_in=channels * 4, ch_out=channels * 8)

        self.zero_pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.last_conv = nn.Conv2d(channels * 8, 1, 4, padding=1, bias=False)

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        x = torch.cat((img_A, img_B), 1)

        x = self.block1(x)
        x = self.Maxpool(x)

        x = self.block2(x)
        x = self.Maxpool(x)

        x = self.block3(x)
        x = self.Maxpool(x)

        x = self.zero_pad(x)

        x = self.last_conv(x)
        x = self.Maxpool(x)
        return x


class GeneratorConvNext001(nn.Module):
    def __init__(self, img_shape=(3, 128, 128), blocks=9, img_ch=3, output_ch=3):
        super().__init__()
        # Initial convolution block
        model = [
            nn.Conv2d(img_ch, 64, 7, stride=1, padding=3, bias=False),
            LayerNorm(64, eps=1e-6, data_format="channels_first"),
        ]

        # Downsampling
        curr_dim = 64
        for _ in range(2):
            model += [
                LayerNorm(curr_dim, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(curr_dim, curr_dim * 2, 4, stride=2, padding=1, bias=False),
                nn.Dropout(0.5),
            ]
            curr_dim *= 2

        for _ in range(blocks):
            model += [ConvNectBlock(curr_dim)]

        # Upsampling
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(
                    curr_dim, curr_dim // 2, 4, stride=2, padding=1, bias=False
                ),
                LayerNorm(curr_dim // 2, eps=1e-6, data_format="channels_first"),
                nn.GELU(),
            ]
            curr_dim = curr_dim // 2

        # Output layer
        model += [nn.Conv2d(curr_dim, output_ch, 7, stride=1, padding=3), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Attention_block2(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.GELU()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttU_ConvNext(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, channels=24):
        super().__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

        self.Conv1 = ConvNext_block(ch_in=img_ch, ch_out=channels)
        self.Conv2 = ConvNext_block(ch_in=channels, ch_out=channels * 2)
        self.Conv3 = ConvNext_block(ch_in=channels * 2, ch_out=channels * 4)
        self.Conv4 = ConvNext_block(ch_in=channels * 4, ch_out=channels * 8)
        self.Conv5 = ConvNext_block(ch_in=channels * 8, ch_out=channels * 16)

        self.Up5 = UpConvNext2(ch_in=channels * 16, ch_out=channels * 8)
        self.Att5 = Attention_block2(
            F_g=channels * 8, F_l=channels * 8, F_int=channels * 4
        )
        self.Up_conv5 = ConvNext_block(ch_in=channels * 16, ch_out=channels * 8)

        self.Up4 = UpConvNext2(ch_in=channels * 8, ch_out=channels * 4)
        self.Att4 = Attention_block2(
            F_g=channels * 4, F_l=channels * 4, F_int=channels * 2
        )
        self.Up_conv4 = ConvNext_block(ch_in=channels * 8, ch_out=channels * 4)

        self.Up3 = UpConvNext2(ch_in=channels * 4, ch_out=channels * 2)
        self.Att3 = Attention_block2(F_g=channels * 2, F_l=channels * 2, F_int=channels)
        self.Up_conv3 = ConvNext_block(ch_in=channels * 4, ch_out=channels * 2)

        self.Up2 = UpConvNext2(ch_in=channels * 2, ch_out=channels)
        self.Att2 = Attention_block2(F_g=channels, F_l=channels, F_int=channels // 2)
        self.Up_conv2 = ConvNext_block(ch_in=channels * 2, ch_out=channels)

        self.Conv_1x1 = nn.Conv2d(
            channels, output_ch, kernel_size=1, stride=1, padding=0
        )
        self.last_activation = nn.Hardtanh()

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x2 = self.dropout(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x3 = self.dropout(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x4 = self.dropout(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        x5 = self.dropout(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = self.last_activation(d1)

        return d1


class Recurrent_block2(nn.Module):
    def __init__(self, ch_out, t=2):
        super().__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv_next = ConvNectBlock(dim=ch_out)

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv_next(x)
            x1 = self.conv_next(x + x1)
        return x1


class RRCNN_block2(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super().__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block2(ch_out, t=t), Recurrent_block2(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class R2U_ConvNext(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, channels=8, t=2):
        super().__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

        self.RRCNN1 = RRCNN_block2(ch_in=img_ch, ch_out=channels, t=t)

        self.RRCNN2 = RRCNN_block2(ch_in=channels, ch_out=channels * 2, t=t)

        self.RRCNN3 = RRCNN_block2(ch_in=channels * 2, ch_out=channels * 4, t=t)

        self.RRCNN4 = RRCNN_block2(ch_in=channels * 4, ch_out=channels * 8, t=t)

        self.RRCNN5 = RRCNN_block2(ch_in=channels * 8, ch_out=channels * 16, t=t)

        self.Up5 = UpConvNext2(ch_in=channels * 16, ch_out=channels * 8)
        self.Up_RRCNN5 = RRCNN_block2(ch_in=channels * 16, ch_out=channels * 8, t=t)

        self.Up4 = UpConvNext2(ch_in=channels * 8, ch_out=channels * 4)
        self.Up_RRCNN4 = RRCNN_block2(ch_in=channels * 8, ch_out=channels * 4, t=t)

        self.Up3 = UpConvNext2(ch_in=channels * 4, ch_out=channels * 2)
        self.Up_RRCNN3 = RRCNN_block2(ch_in=channels * 4, ch_out=channels * 2, t=t)

        self.Up2 = UpConvNext2(ch_in=channels * 2, ch_out=channels)
        self.Up_RRCNN2 = RRCNN_block2(ch_in=channels * 2, ch_out=channels, t=t)

        self.Conv_1x1 = nn.Conv2d(
            channels, output_ch, kernel_size=1, stride=1, padding=0
        )
        self.last_activation = nn.Hardtanh()

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = self.last_activation(d1)

        return d1


class R2AttU_ConvNext(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, channels=8, t=2):
        super().__init__()

        self.Maxpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block2(ch_in=img_ch, ch_out=channels, t=t)

        self.RRCNN2 = RRCNN_block2(ch_in=channels, ch_out=channels * 2, t=t)

        self.RRCNN3 = RRCNN_block2(ch_in=channels * 2, ch_out=channels * 4, t=t)

        self.RRCNN4 = RRCNN_block2(ch_in=channels * 4, ch_out=channels * 8, t=t)

        self.RRCNN5 = RRCNN_block2(ch_in=channels * 8, ch_out=channels * 16, t=t)

        self.Up5 = UpConvNext2(ch_in=channels * 16, ch_out=channels * 8)
        self.Att5 = Attention_block2(
            F_g=channels * 8, F_l=channels * 8, F_int=channels * 4
        )
        self.Up_RRCNN5 = RRCNN_block2(ch_in=channels * 16, ch_out=channels * 8, t=t)

        self.Up4 = UpConvNext2(ch_in=channels * 8, ch_out=channels * 4)
        self.Att4 = Attention_block2(
            F_g=channels * 4, F_l=channels * 4, F_int=channels * 2
        )
        self.Up_RRCNN4 = RRCNN_block2(ch_in=channels * 8, ch_out=channels * 4, t=t)

        self.Up3 = UpConvNext2(ch_in=channels * 4, ch_out=channels * 2)
        self.Att3 = Attention_block2(F_g=channels * 2, F_l=channels * 2, F_int=channels)
        self.Up_RRCNN3 = RRCNN_block2(ch_in=channels * 4, ch_out=channels * 2, t=t)

        self.Up2 = UpConvNext2(ch_in=channels * 2, ch_out=channels)
        self.Att2 = Attention_block2(F_g=channels, F_l=channels, F_int=channels // 2)
        self.Up_RRCNN2 = RRCNN_block2(ch_in=channels * 2, ch_out=channels, t=t)

        self.Conv_1x1 = nn.Conv2d(
            channels, output_ch, kernel_size=1, stride=1, padding=0
        )
        self.last_activation = nn.Hardtanh()

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = self.last_activation(d1)

        return d1


if __name__ == "__main__":
    device = torch.device("cuda")
    input_tensor = torch.rand(1, 3, 512, 512, device=device)
    model = R2AttU_ConvNext()
    model.to(device)
    output_tensor = model.forward(input_tensor)
    print(output_tensor)
