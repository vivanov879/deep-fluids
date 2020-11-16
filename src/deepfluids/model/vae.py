""" An implementation of Variational Autoencoder model """

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


class VAE(BaseLightningModel):
    def __init__(self, z_num: int):
        super().__init__()
        self.encoder = EncoderModel(32)
        self.generator = GeneratorModel(z_num)
        self.fc1 = Linear(32, z_num)
        self.fc2 = Linear(32, z_num)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enc = self.encoder(x)
        enc = F.leaky_relu(enc)
        mu = self.fc1(enc)
        logvar = self.fc2(enc)
        z = self._reparameterize(mu, logvar)
        x = self.generator(z)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return z, x, KLD

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _train_valid_helper(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """A helper function that extracts samples from a batch of data, runs a forward step and calculates loss

        Args:
          batch: input batch

        Returns:
          a tuple of ground truth masks, prediction, a loss value, and f1 score

        """
        x = batch['x']
        y = batch['y']

        p_num = 2

        z, G_s, KLD = self(x)
        _, G_ = jacobian3(G_s)
        G_jaco_, G_vort_ = jacobian3(G_)
        x_jaco, x_vort = jacobian3(x)

        # _logger.info(f"{G_.shape=}")
        # _logger.info(f"{x.shape=}")
        # _logger.info(f"{G_jaco_.shape=}")
        # _logger.info(f"{x_jaco.shape=}")
        # _logger.info(f"{z[:, -p_num:].shape=}, {y.shape=}")

        loss = KLD + F.l1_loss(G_, x) + F.l1_loss(G_jaco_, x_jaco) + F.mse_loss(z[:, -p_num:], y)

        return loss
