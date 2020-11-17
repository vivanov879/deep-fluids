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
    checkpoint_path = "/home/vivanov/Projects/deep-fluids/experiments/NN/version_6/checkpoints/epoch=177.ckpt"
    model = NNModel.load_from_checkpoint(checkpoint_path=checkpoint_path).cuda()
    model.eval()

    p_num = 2
    sim_idx = 0
    num_sims, num_frames, data = extract_sim_from_code16(sim_idx)

    _logger.info(f"{data['x'].shape=}, {num_frames=}")

    x = data['x'][[0]]
    xs = []

    with torch.no_grad():
        for i in tqdm(range(num_frames)):
            dp = data['dp'][[i]]

            p = data['dp'][[i]]

            xs.append(np.concatenate([x, p], 1))
            input = np.concatenate([x, p, dp], 1)
            input = torch.Tensor(input).cuda()

            dx = model(input).cpu().numpy()

            x += dx
            _logger.info(f"{dx=}")


    xs = np.array(xs, np.float32)

    _logger.info(f"{xs.shape=}")

    xs_ps_fn = Path("/Users/vivanov/Projects/deep-fluids/experiments/NN/xs_ps_inference.npz")
    np.savez_compressed(xs_ps_fn, x=xs)


if __name__ == '__main__':
    main()
