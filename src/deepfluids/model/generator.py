""" Utils for creating training models"""
from typing import Dict, List

import torch
import pytorch_lightning as pl
from torch.nn import Linear, ModuleList, Conv3d
import torch.nn.functional as F
from loguru import logger as _logger

from .base import BaseLightningModel
from ..model.utils import jacobian3


class GeneratorModel(BaseLightningModel):
    def __init__(self):
        super().__init__()
        self.fc = Linear(3, 128 * 4 * 8 * 14)
        self.repeat_num = 4
        self.num_conv = 4
        self.filters = 128
        self.convs = ModuleList()
        for _ in range(self.repeat_num):
            for _ in range(self.num_conv):
                layer = Conv3d(self.filters, self.filters, kernel_size=3, padding=1)
                self.convs.append(layer)
        self.last_conv = Conv3d(self.filters, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x.reshape(-1, 3)
        x = self.fc(x)
        x = x.reshape(-1, 128, 4, 8, 14)
        x0 = x

        for idx in range(self.repeat_num):
            for conv_idx in range(self.num_conv):

                lookup_idx = idx * self.num_conv + conv_idx
                layer = self.convs[lookup_idx]
                x = layer(x)
                x = F.leaky_relu(x)

            if idx < self.repeat_num - 1:
                x += x0
                x = F.interpolate(x, scale_factor=2)
                x0 = x

            else:
                x += x0

        x = self.last_conv(x)

        return x

    def _train_valid_helper(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:

        x = batch['x']
        y = batch['y']

        G_s = self(y)
        _, G_ = jacobian3(G_s)
        G_jaco_, G_vort_ = jacobian3(G_)
        x_jaco, x_vort = jacobian3(x)

        _logger.info(f"{G_.shape=}")
        _logger.info(f"{x.shape=}")
        _logger.info(f"{G_jaco_.shape=}")
        _logger.info(f"{x_jaco.shape=}")
        loss = F.l1_loss(G_, x) + F.l1_loss(G_jaco_, x_jaco)

        return loss
