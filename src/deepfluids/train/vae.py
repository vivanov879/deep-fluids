""" A script to train a VAE model """

from pathlib import Path

from src.deepfluids.model.vae import VAE
from .base import train
from ..dataset.autoencoder.train import AutoencoderDataset
from ..model.autoencoder import Autoencoder

if __name__ == '__main__':
    model = VAE(16)
    data_dir = Path("/Users/vivanov/Projects/deep-fluids/data/smoke3_mov200_f400/v")
    dataset = AutoencoderDataset(data_dir)
    train(model, dataset, "VAE")
