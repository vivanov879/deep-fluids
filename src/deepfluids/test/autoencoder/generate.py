""" A script to generate velocity vector field from a latent code produced by a Neural Network """
from pathlib import Path
import os

import numpy as np
import torch
from tqdm import tqdm
import click
from src.deepfluids.model.autoencoder import Autoencoder
from ..utils import extract_sim_from_code16
from ...model.utils import jacobian3
from ...model.generator import GeneratorModel
from loguru import logger as _logger


@click.command()
@click.option('--nn_inference', is_flag=True, help='Use nn inference')
def main(nn_inference):
    sim_idx = 0

    if nn_inference:
        _logger.info(f"Using latent code inferenced by nn")
        xs_ps_fn = Path("/Users/vivanov/Projects/deep-fluids/experiments/NN/xs_ps_inference.npz")
        data = np.load(xs_ps_fn)
        data = {
            'x': np.asarray(data['x'], np.float32)
        }

    else:
        num_sims, num_frames, data = extract_sim_from_code16(sim_idx)

    checkpoint_path = "/Users/vivanov/Projects/deep-fluids/experiments/Autoencoder/version_0/checkpoints/epoch=5.ckpt"
    model = Autoencoder.load_from_checkpoint(z_num=16, checkpoint_path=checkpoint_path).cuda()
    model = model.generator
    model.eval()

    batch_size = 5
    niter = len(data['x']) // batch_size
    niter += 1

    G = []
    with torch.no_grad():
        for b in tqdm(range(niter), desc="Iterating over z"):
            z = data['x'][batch_size * b:batch_size * (b + 1), :]
            if len(z) == 0:
                continue
            G_s = model(torch.Tensor(z).cuda())
            _, G_ = jacobian3(G_s)

            G_ = G_.permute(0, 2, 3, 4, 1)
            G_ = G_.cpu().numpy()

            G.append(G_)

    G = np.concatenate(G, axis=0)

    out_dir = Path("/Users/vivanov/Projects/deep-fluids/experiments/Autoencoder/velocity_field/")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i, G_ in enumerate(tqdm(G, desc="Enumerating G")):
        v_path_ = Path("/home/vivanov/Projects/deep-fluids/data/smoke3_mov200_f400/v") / f"{sim_idx}_{i}.npz"
        dump_path = os.path.join(out_dir, '%d.npz' % i)
        np.savez_compressed(dump_path, x=G_, y=np.load(v_path_)['y'])


if __name__ == '__main__':
    main()
