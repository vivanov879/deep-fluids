""" Utils for training a neural network over the latent space code """

from typing import Dict, List

import torch
import pytorch_lightning as pl
from torch.nn import Linear, ModuleList, Conv3d, BatchNorm1d, Dropout
import torch.nn.functional as F
from loguru import logger as _logger

from .base import BaseLightningModel
from ..model.utils import jacobian3

class NNModel(BaseLightningModel):
    def __init__(self):
        super().__init__()
        self.z_num = 16
        self.p_num = 2
        self.hidden_size = 256
        self.window = 30
        self.fc1 = Linear(self.z_num + self.p_num, self.hidden_size)
        self.bn1 = BatchNorm1d(self.hidden_size)
        self.do1 = Dropout(0.1)
        self.fc2 = Linear(self.hidden_size, self.hidden_size)
        self.bn2 = BatchNorm1d(self.hidden_size)
        self.do2 = Dropout(0.1)
        self.fc3 = Linear(self.hidden_size, self.z_num - self.p_num)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.bn1(x)
        x = self.do1(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.bn2(x)
        x = self.do2(x)
        x = self.fc3(x)
        x = F.tanh(x)

        return x

    def _train_valid_helper(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch['x'][:, 0, :]
        loss = 0
        for i in range(self.window):
            p = batch['p'][:, i, :]
            dp = batch['dp'][:, i, :]
            input = torch.cat([x, p, dp], dim=1)
            dx_pred = self(input)
            loss += F.mse_loss(dx_pred, batch['y'][:, i, :])

            x += dx_pred

        return loss
