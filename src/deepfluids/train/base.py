""" An implementation of a base training script pattern """

from pathlib import Path

import pytorch_lightning as pl
from loguru import logger as _logger
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, random_split, Dataset

from ..dataset.base import BaseDataset
from ..dataset.generator import GeneratorDataset
from ..model.generator import GeneratorModel
import horovod.torch as hvd

def train(model: pl.LightningModule, dataset: Dataset, experiment: str):
    hvd.init()
    pl.seed_everything(215)

    _logger.info(f"Initializing a model")

    _logger.info(f"Initializing dataset")

    dataset_train, dataset_val = random_split(dataset,
                                              [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))])

    dataloader_train = DataLoader(dataset_train, batch_size=8, pin_memory=True, shuffle=True, num_workers=8)
    dataloader_val = DataLoader(dataset_val, batch_size=8, pin_memory=True, shuffle=False, num_workers=8)

    experiments_path = Path("/Users/vivanov/Projects/deep-fluids/experiments/")

    tb_logger = pl_loggers.TensorBoardLogger(str(experiments_path), name=experiment)

    # initialize a trainer
    trainer = pl.Trainer(
        gpus=1,
        progress_bar_refresh_rate=2,
        default_root_dir=experiments_path / experiment,
        max_epochs=1000,
        logger=tb_logger,
        distributed_backend='horovod',
        # replace_sampler_ddp=False,
    )

    trainer.fit(model, dataloader_train, dataloader_val)

