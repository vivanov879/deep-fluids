""" A script to test a trained Generator model """
import os

import numpy as np
import torch
from tqdm import tqdm

from src.deepfluids.model.utils import jacobian3
from ..model.generator import GeneratorModel
from loguru import logger as _logger

def main():
    checkpoint_path = "/home/vivanov/Projects/deep-fluids/experiments/Generator/version_0/checkpoints/epoch=32.ckpt"
    model = GeneratorModel.load_from_checkpoint(checkpoint_path=checkpoint_path).cuda()
    model.eval()

    p1, p2 = 10, 2

    # eval
    y1 = 5
    y2 = 3
    y3 = 250

    batch_size = 5

    niter = int(y3 / batch_size)

    c1 = p1 / float(y1 - 1) * 2 - 1
    c2 = p2 / float(y2 - 1) * 2 - 1
    c_num = 3

    z_range = [-1, 1]
    z_varying = np.linspace(z_range[0], z_range[1], num=y3)
    z_shape = (y3, c_num)

    z_c = np.zeros(shape=z_shape)
    z_c[:, 0] = c1
    z_c[:, 1] = c2
    z_c[:, -1] = z_varying

    G = []
    with torch.no_grad():
        for b in tqdm(range(niter), desc="Iterating over z"):
            z = z_c[batch_size * b:batch_size * (b + 1), :]

            G_s = model(torch.Tensor(z).cuda())
            _, G_ = jacobian3(G_s)

            G_ = G_.permute(0,2,3,4,1)
            G_ = G_.cpu().numpy()
            _logger.info(f"{G_[0][0][0][0]=}")
            _logger.info(f"{G_.shape=}")

            G.append(G_)
    G = np.concatenate(G, axis=0)

    # save
    title = '%d_%d' % (p1, p2)
    out_dir = os.path.join("out/", title)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i, G_ in tqdm(enumerate(G), desc="Enumerating G"):
        dump_path = os.path.join(out_dir, '%d.npz' % i)
        np.savez_compressed(dump_path, x=G_)


if __name__ == '__main__':
    main()
