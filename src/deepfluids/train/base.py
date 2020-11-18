""" An implementation of a base training script pattern """

from pathlib import Path

import pytorch_lightning as pl
from loguru import logger as _logger
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, random_split, Dataset, Subset

from ..dataset.base import BaseDataset
from ..dataset.generator import GeneratorDataset
from ..model.generator import GeneratorModel
import horovod.torch as hvd


def train(model: pl.LightningModule, dataset: Dataset, experiment: str, batch_size: int = 8,
          first_ten_percent_val=False):
    """
    A method that trains a model
    Args:
        model: model to train
        dataset: dataset for training
        experiment: experiment name
        batch_size: batch size
        first_ten_percent_val: a flag indicating that first portion of a dataset is to be set aside as validation

    Returns:

    """
    hvd.init()
    pl.seed_everything(215)

    _logger.info(f"Initializing a model")

    _logger.info(f"Initializing dataset")

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - int(0.9 * len(dataset))

    if first_ten_percent_val:
        dataset_val = Subset(dataset, range(val_size))
        dataset_train = Subset(dataset, range(val_size, train_size))
    else:
        dataset_train, dataset_val = random_split(dataset, [train_size, val_size])
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=8)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=8)

    experiments_path = Path("/Users/vivanov/Projects/deep-fluids/experiments/")

    tb_logger = pl_loggers.TensorBoardLogger(str(experiments_path), name=experiment)

    # initialize a trainer
    trainer = pl.Trainer(
        gpus=1,
        progress_bar_refresh_rate=2,
        default_root_dir=experiments_path / experiment,
        max_epochs=9999999,
        logger=tb_logger,
        distributed_backend='horovod',
        # replace_sampler_ddp=False,
    )

    trainer.fit(model, dataloader_train, dataloader_val)
