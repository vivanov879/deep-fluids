""" An implementation of a dataset designed for neural network latent code prediction """

from pathlib import Path
from typing import List, Dict, NamedTuple

from torch.utils.data import DataLoader, Dataset
import numpy as np
from loguru import logger as _logger
from tqdm import tqdm
import pickle


class StartPoint(NamedTuple):
    """
    Stores the information on the sequence start
    """
    sim_idx: int
    frame_idx: int


class NeuralNetworkDataset(Dataset):
    def __init__(self, data_fn: Path):
        super().__init__()

        self.data = pickle.load(open(data_fn, "rb"))
        self.num_sims = len(self.data)
        self.window = 30

        self.num_frames = len(self.data[0]['x'])
        self.start_points = self._build_start_point_index()

    def _build_start_point_index(self):
        start_points = []
        for sim_idx in range(self.num_sims):
            for frame_idx in range(self.num_frames - self.window):
                start_point = StartPoint(sim_idx=sim_idx, frame_idx=frame_idx)
                start_points.append(start_point)
        return start_points

    def __len__(self):
        return len(self.start_points)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        start_point = self.start_points[idx]

        data = self.data[start_point.sim_idx]

        data = {
            key: np.array(value[start_point.frame_idx: start_point.frame_idx + self.window], dtype=np.float32) for
            key, value in data.items()
        }

        data['y'] -= data['x']
        return data


if __name__ == '__main__':
    data_dir = Path("/Users/vivanov/Projects/deep-fluids/experiments/Autoencoder/code16.pickle")
    dataset = NeuralNetworkDataset(data_dir)
    entry = dataset[100]
    _logger.info(f"{entry['x'].shape=}")
    _logger.info(f"{entry['y'].shape=}")

    dataloader = DataLoader(dataset, batch_size=4, pin_memory=True)
    for entry in dataloader:
        _logger.info(f"{entry.keys()=}")
        _logger.info(f"{entry['x'].shape=}")
        _logger.info(f"{entry['y'].shape=}")
        _logger.info(f"{entry['p'].shape=}")
        break
