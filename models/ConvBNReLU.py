import torch.nn as nn
import math

from torchsummary import summary


import torch
from torch.quantization import QuantStub, DeQuantStub


class ConvBNReLU(nn.Sequential):
    def __init__(
        self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, groups=1
    ):

        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False
            ),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            # Replace with ReLU
            nn.ReLU(inplace=False)
        )
