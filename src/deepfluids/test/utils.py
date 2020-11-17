""" Utils for testing trained models """
import pickle
from pathlib import Path

import numpy as np


def extract_sim_from_code16(sim_idx: int):
    data_fn = Path("/Users/vivanov/Projects/deep-fluids/experiments/Autoencoder/code16.pickle")

    data = pickle.load(open(data_fn, "rb"))

    num_sims = len(data)
    num_frames = len(data[0]['x'])

    data = data[sim_idx]

    data = {
        key: np.array(value, dtype=np.float32) for key, value in data.items()
    }
    data['y'] -= data['x']

    return num_sims, num_frames, data
