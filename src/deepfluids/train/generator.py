""" A script to train a generator model """
from pathlib import Path

import pytorch_lightning as pl
from loguru import logger as _logger
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, random_split

from .base import train
from ..dataset.generator import GeneratorDataset
from ..model.generator import GeneratorModel
import horovod.torch as hvd

if __name__ == '__main__':
    model = GeneratorModel(3)
    data_dir = Path("/Users/vivanov/Projects/deep-fluids/data/smoke3_vel5_buo3_f250/v")

    dataset = GeneratorDataset(data_dir)
    train(model, dataset, "Generator")
