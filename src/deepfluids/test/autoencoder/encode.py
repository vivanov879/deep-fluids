""" A script to generate latent codes using an Autoencoder """
from pathlib import Path

import torch
from loguru import logger as _logger
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.deepfluids.dataset.autoencoder.infer import AutoencoderInferenceDataset
from src.deepfluids.model.autoencoder import Autoencoder


def dump_latent_codes():

    checkpoint_path = "/home/vivanov/Projects/deep-fluids/experiments/Autoencoder/version_0/checkpoints/epoch=5.ckpt"
    model = Autoencoder.load_from_checkpoint(z_num = 16, checkpoint_path=checkpoint_path).cuda()
    model.eval()

    n_path = Path("/Users/vivanov/Projects/deep-fluids/data/smoke3_mov200_f400/n.npz")
    with np.load(n_path) as data:
        nx = data['nx']
        nz = data['nz']

    num_sims = nx.shape[0]
    num_frames = nx.shape[1]
    _logger.info(f"num_sims: {num_sims}, num_frames: {num_frames}")

    dx_list = (nx[:, 1:] - nx[:, :-1]).reshape([-1, 1])
    dz_list = (nz[:, 1:] - nz[:, :-1]).reshape([-1, 1])
    dp_list = np.concatenate((dx_list, dz_list), axis=-1)


    x_list = (nx[:, :-1]).reshape([-1, 1])
    z_list = (nz[:, :-1]).reshape([-1, 1])
    p_list = np.concatenate((x_list, z_list), axis=-1)
    assert p_list.shape == dp_list.shape

    c_list = []
    p_num = 2

    data_dir = Path("/Users/vivanov/Projects/deep-fluids/data/smoke3_mov200_f400/v")
    dataset = AutoencoderInferenceDataset(data_dir, num_frames)
    dataloader = DataLoader(dataset, batch_size=8, pin_memory=True, shuffle=False, num_workers=8)

    _logger.info(f"{len(dataset)=}")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            x = batch['x'].cuda()
            c, _ = model(x)
            c = c.cpu().numpy()
            c = c[:, :-p_num]
            c_list.append(c)

    c_list = np.concatenate(c_list)
    x_list = []
    y_list = []

    for i in range(num_sims):
        s1 = i * num_frames
        s2 = (i + 1) * num_frames
        x_list.append(c_list[s1:s2 - 1, :])
        y_list.append(c_list[s1 + 1:s2, :])

    x_list = np.concatenate(x_list)
    y_list = np.concatenate(y_list)
    _logger.info(f"{x_list.shape=}, {y_list.shape=}, {dp_list.shape=} {p_list.shape=}")

    code_path = Path("/Users/vivanov/Projects/deep-fluids/experiments/Autoencoder/") / "code16.npz"
    np.savez_compressed(code_path,
                        x=x_list,
                        y=y_list,
                        dp=dp_list,
                        p=p_list,
                        s=num_sims,
                        f=num_frames)

if __name__ == '__main__':
    dump_latent_codes()
