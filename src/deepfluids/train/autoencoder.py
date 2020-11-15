""" A script to train an Autoencoder model """

from pathlib import Path

from .base import train
from ..dataset.autoencoder.train import AutoencoderDataset
from ..model.autoencoder import Autoencoder

if __name__ == '__main__':
    model = Autoencoder(16)
    data_dir = Path("/Users/vivanov/Projects/deep-fluids/data/smoke3_mov200_f400/v")
    dataset = AutoencoderDataset(data_dir)
    train(model, dataset, "Autoencoder")
