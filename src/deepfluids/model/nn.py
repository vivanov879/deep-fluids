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
        self.hidden_size = 512
        self.window = 30
        self.fc1 = Linear(self.z_num + self.p_num, self.hidden_size)
        self.fc2 = Linear(self.hidden_size, self.hidden_size)
        self.fc3 = Linear(self.hidden_size, self.z_num - self.p_num)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.sin(1 + x)
        x = self.fc2(x)
        x = torch.sin(1 + x)
        x = self.fc3(x)
        x = torch.sin(1 + x)

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
            #_logger.info(f"{dx_pred[19]=}")
            #_logger.info(f"{batch['y'][:, i, :][19]=}")
            x += dx_pred

        return loss
