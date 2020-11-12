""" An implementation of base lightning model """

from typing import Dict, List

import pytorch_lightning as pl
import torch


class BaseLightningModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def _train_valid_helper(self, batch: Dict[str, torch.Tensor]):
        raise NotImplemented

    def forward(self, x: torch.Tensor):
        raise NotImplemented

    def training_step(self, batch: Dict[str, torch.Tensor], batch_nb) -> torch.Tensor:
        loss = self._train_valid_helper(batch)
        self.log_dict({"loss/train": loss.item()})
        return loss

    def validation_step(self, batch: Dict[str, torch.ByteTensor], batch_nb) -> torch.Tensor:
        loss = self._train_valid_helper(batch)
        return loss

    def validation_epoch_end(self, outputs: List[torch.Tensor]):
        # OPTIONAL
        avg_loss = torch.stack([loss for loss in outputs]).mean()
        self.log_dict({"loss/val": avg_loss})

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """A method that returns an optimizer used for training.

        Args:

        Returns:
            torch optimizer

        """
        return torch.optim.Adam(self.parameters(), lr=1e-4)
