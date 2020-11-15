""" An implementation of a dataset designed for neural network latent code prediction """


from pathlib import Path
from typing import List, Dict

from torch.utils.data import DataLoader, Dataset
import numpy as np
from loguru import logger as _logger
from tqdm import tqdm


class NeuralNetworkDataset(Dataset):
    def __init__(self, data_fn: Path):
        super().__init__()

        data_fn = str(data_fn)
        self.data = np.load(data_fn)
        self.data = {
            'x': np.asarray(self.data['x'], np.float32),
            'y': np.asarray(self.data['y'], np.float32),
            'p': np.asarray(self.data['p'], np.float32)
        }

        self.window = 5
        self.num_sims = self.data['s']
        self.seq_len = len(self.data['x']) // self.num_sims
        self.start_idxs = self._build_start_point_index()

    def _build_start_point_index(self):
        start_idxs = []
        for i in range(len(self.data['x'])):
            if i % self.seq_len in list(range(self.seq_len + 1))[-6:]:
                continue
            start_idxs.append(i)
        return start_idxs

    def __len__(self):
        return len(self.start_idxs)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        start_idx = self.start_idxs[idx]
        data = {
            'x': np.asarray(self.data['x'][start_idx: start_idx + self.window], dtype=np.float32),
            'y': np.asarray(self.data['y'][start_idx: start_idx + self.window], dtype=np.float32),
            'p': np.asarray(self.data['p'][start_idx: start_idx + self.window], dtype=np.float32)
        }
        data['y'] -= data['x']
        return data

if __name__ == '__main__':
    data_dir = Path("/Users/vivanov/Projects/deep-fluids/experiments/Autoencoder/code16.npz")
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






