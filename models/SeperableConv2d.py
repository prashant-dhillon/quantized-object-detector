from torch.nn import Conv2d, Sequential, BatchNorm2d
from models.ConvBNReLU import ConvBNReLU

from torch import nn


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(SeparableConv2d, self).__init__()
        self.convbnrelu = ConvBNReLU(
            in_planes=in_channels,
            out_planes=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels
        )
        self.conv2 = Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        x = self.convbnrelu(x)
        x = self.conv2(x)
        return x


def get_seperable_conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    print(f'in_channels = {in_channels}')
    print(f'out_channels = {out_channels}')
    print(f'kernel_size = {kernel_size}')
    print(f'stride = {stride}')
    print(f'padding = {padding}')
    return Sequential(
        Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
        ),
        BatchNorm2d(in_channels),
        nn.ReLU6(),
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )
