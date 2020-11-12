""" Utils for creating training models"""
from typing import Dict, List

import torch
import pytorch_lightning as pl
from torch.nn import Linear, ModuleList, Conv3d
import torch.nn.functional as F
from loguru import logger as _logger
from ..model.utils import jacobian3


class EncoderModel(torch.nn.Module):
    def __init__(self, z_num: int):
        super().__init__()
        self.repeat_num = 4
        self.num_conv = 4
        self.filters = 64
        self.z_num = z_num

        self.conv1 = Conv3d(3, self.filters, kernel_size=3, padding=1)

        self.convs = ModuleList()
        for repeat_idx in range(self.repeat_num):
            for conv_idx in range(self.num_conv):
                if conv_idx == self.num_conv - 1:
                    if repeat_idx == 0:
                        layer = Conv3d(self.filters + 3, self.filters, kernel_size=3, padding=1)
                    else:
                        layer = Conv3d(2 * self.filters, self.filters, kernel_size=3, padding=1)
                else:
                    layer = Conv3d(self.filters, self.filters, kernel_size=3, padding=1)
                self.convs.append(layer)

        self.conv_last = Conv3d(self.filters, self.filters, kernel_size=3, stride=2, padding=1)
        self.fc = Linear(self.filters * 6 * 9 * 6, self.z_num)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        batch_size = x.shape[0]

        x0 = x

        x = self.conv1(x)

        for idx in range(self.repeat_num):
            for conv_idx in range(self.num_conv):
                if conv_idx == self.num_conv - 1:
                    x = torch.cat([x, x0], dim=1)
                lookup_idx = idx * self.num_conv + conv_idx
                layer = self.convs[lookup_idx]
                x = layer(x)
                x = F.leaky_relu(x)

            if idx < self.repeat_num - 1:
                x = self.conv_last(x)
                x0 = x

        x = x.view(batch_size, -1)
        x = self.fc(x)

        return x
