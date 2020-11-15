""" A script to test inference of a neural network over latent space code """
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import extract_sim_from_code16
from ..dataset.nn import NeuralNetworkDataset
from ..model.nn import NNModel
import numpy as np
from loguru import logger as _logger


def main():
    checkpoint_path = "/home/vivanov/Projects/deep-fluids/experiments/NN/version_0/checkpoints/epoch=7.ckpt"
    model = NNModel.load_from_checkpoint(checkpoint_path=checkpoint_path).cuda()
    model.eval()

    sim_idx = 0
    num_sims, num_frames, data = extract_sim_from_code16(sim_idx)

    x = data['x'][0]
    x = x[None, ...]
    xs = []

    with torch.no_grad():
        for i in tqdm(range(num_frames)):
            p = data['p'][i]
            p = p[None, ...]

            input = np.concatenate([x, p], 1)
            input = torch.Tensor(input).cuda()

            dx = model(input).cpu().numpy()

            x += dx

            xs.append(x.flatten())

    xs = np.asarray(xs, np.float32)

    _logger.info(f"{xs.shape=}")

    xs_ps_fn = Path("/Users/vivanov/Projects/deep-fluids/experiments/NN/xs_ps_inference.npz")
    np.savez_compressed(xs_ps_fn, x=xs)


if __name__ == '__main__':
    main()
