import torch.nn as nn
from models.ConvBNReLU import ConvBNReLU


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(
                ConvBNReLU(inp, hidden_dim, kernel_size=1, stride=1, padding=0)
            )

        layers.extend(
            [
                # dw
                ConvBNReLU(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=hidden_dim
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup, momentum=0.1),
            ]
        )
        self.conv = nn.Sequential(*layers)
        # Replace torch.add with floatfunctional
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_res_connect:
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)
