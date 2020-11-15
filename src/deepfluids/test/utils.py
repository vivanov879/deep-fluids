""" Utils for testing trained models """
from pathlib import Path

import numpy as np


def extract_sim_from_code16(sim_idx: int):
    data_dir = Path("/Users/vivanov/Projects/deep-fluids/experiments/Autoencoder/code16.npz")

    data = np.load(data_dir)
    num_sims = data['s']
    data = {'x': np.asarray(data['x'], np.float32),
            'p': np.asarray(data['p'], np.float32),
            'y': np.asarray(data['y'], np.float32)}

    num_frames = len(data['x']) // num_sims
    data['x'] = data['x'][sim_idx * num_frames: (sim_idx + 1) * num_frames]
    data['p'] = data['p'][sim_idx * num_frames: (sim_idx + 1) * num_frames]

    return num_sims, num_frames, data
