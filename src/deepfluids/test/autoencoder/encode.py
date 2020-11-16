""" A script to generate latent codes using an Autoencoder """
from pathlib import Path

import torch
from loguru import logger as _logger
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle

from src.deepfluids.dataset.autoencoder.infer import AutoencoderInferenceDataset
from src.deepfluids.model.autoencoder import Autoencoder


def dump_latent_codes():
    checkpoint_path = "/home/vivanov/Projects/deep-fluids/experiments/Autoencoder/version_1/checkpoints/epoch=7.ckpt"
    model = Autoencoder.load_from_checkpoint(z_num=16, checkpoint_path=checkpoint_path).cuda()
    model.eval()

    n_path = Path("/Users/vivanov/Projects/deep-fluids/data/smoke3_mov200_f400/n.npz")
    with np.load(n_path) as data:
        nx = data['nx']
        nz = data['nz']

    num_sims = nx.shape[0]
    num_frames = nx.shape[1]

    _logger.info(f"num_sims: {num_sims}, num_frames: {num_frames}")

    dx_list = (nx[:, 1:] - nx[:, :-1])[..., None]
    dz_list = (nz[:, 1:] - nz[:, :-1])[..., None]
    dp_list = np.concatenate((dx_list, dz_list), axis=-1)

    x_list = nx[:, :-1][..., None]
    z_list = nz[:, :-1][..., None]
    p_list = np.concatenate((x_list, z_list), axis=-1)
    assert p_list.shape == dp_list.shape

    _logger.info(f"{p_list.shape=}")

    res = []
    for sim_idx in range(num_sims):
        res.append(
            {
                'p': p_list[sim_idx],
                'dp': dp_list[sim_idx]
            }
        )

    p_num = 2

    data_dir = Path("/Users/vivanov/Projects/deep-fluids/data/smoke3_mov200_f400/v")

    for sim_idx in tqdm(range(num_sims)):
        c_list = []
        dataset = AutoencoderInferenceDataset(data_dir, sim_idx, num_frames)
        dataloader = DataLoader(dataset, batch_size=8, pin_memory=True, shuffle=False, num_workers=8)
        for i, batch in enumerate(tqdm(dataloader)):
            with torch.no_grad():
                x = batch['x'].cuda()
                c, _ = model(x)
                c = c.cpu().numpy()
                c = c[:, :-p_num]
                c_list.append(c)

        c_list = np.concatenate(c_list, 0)

        res[sim_idx]['x'] = c_list[:-1]
        res[sim_idx]['y'] = c_list[1:]

    code_path = Path("/Users/vivanov/Projects/deep-fluids/experiments/Autoencoder/") / "code16.pickle"
    pickle.dump(res, open(code_path, "wb"))


if __name__ == '__main__':
    dump_latent_codes()
