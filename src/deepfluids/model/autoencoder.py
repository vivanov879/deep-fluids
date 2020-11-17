""" An implementation of Autoencoder model """

from typing import Dict, List, Tuple

import torch
import pytorch_lightning as pl
from torch.nn import Linear, ModuleList, Conv3d
import torch.nn.functional as F
from loguru import logger as _logger

from .base import BaseLightningModel
from .encoder import EncoderModel
from .generator import GeneratorModel
from ..model.utils import jacobian3


class Autoencoder(BaseLightningModel):
    def __init__(self, z_num: int):
        """
        Autoencoder model
        Args:
            z_num: latent code dimension
        """
        super().__init__()
        self.encoder = EncoderModel(z_num)
        self.generator = GeneratorModel(z_num)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x = self.generator(z)

        return z, x

    def _train_valid_helper(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """A helper function that extracts samples from a batch of data, runs a forward step and calculates loss

        Args:
          batch: input batch

        Returns:
            loss

        """
        x = batch['x']
        y = batch['y']

        p_num = 2

        z, G_s = self(x)
        _, G_ = jacobian3(G_s)
        G_jaco_, G_vort_ = jacobian3(G_)
        x_jaco, x_vort = jacobian3(x)

        loss = F.l1_loss(G_, x) + F.l1_loss(G_jaco_, x_jaco) + F.mse_loss(z[:, -p_num:], y)

        return loss
