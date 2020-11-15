""" A script to train a neural network over a latent space code """


from pathlib import Path

import pytorch_lightning as pl
from loguru import logger as _logger
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, random_split

from ..dataset.nn import NeuralNetworkDataset
from ..model.nn import NNModel
from .base import train
from ..dataset.generator import GeneratorDataset
from ..model.generator import GeneratorModel
import horovod.torch as hvd

if __name__ == '__main__':
    model = NNModel()
    data_dir = Path("/Users/vivanov/Projects/deep-fluids/experiments/Autoencoder/code16.npz")

    dataset = NeuralNetworkDataset(data_dir)
    train(model, dataset, "NN")
