""" Autoencoder dataset """
import os
from pathlib import Path
from typing import Dict

import numpy as np
from loguru import logger as _logger
from torch.utils.data import DataLoader, Dataset

from ...dataset.base import BaseDataset


class AutoencoderInferenceDataset(Dataset):
    def __init__(self, data_dir: Path, sim_idx: int, num_frames: int):
        super().__init__()
        self.data_dir = data_dir
        self.num_frames = num_frames
        self.sim_idx = sim_idx

    def get_fn(self, frame_idx: int):
        """
        Returns file name given an index
        Args:
            frame_idx: frame index

        Returns: filename

        """
        fn = self.data_dir / f"{self.sim_idx}_{frame_idx}.npz"
        return fn

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        fn = self.get_fn(idx)
        data = self.fn2data(fn)
        return data

    def fn2data(self, fn: Path) -> Dict[str, np.ndarray]:
        """
        Extracts data for a file
        Args:
            fn: filename

        Returns: dictionary with data

        """
        data = np.load(fn)
        data = {
            'x': np.array(data['x'].transpose(3, 0, 1, 2), dtype=np.float32),  # CDHW
            'y': np.array(data['y'][:, -1], dtype=np.float32)
        }
        return data

    def __len__(self):
        return self.num_frames


if __name__ == '__main__':
    data_dir = Path("/Users/vivanov/Projects/deep-fluids/data/smoke3_mov200_f400/v")
    dataset = AutoencoderInferenceDataset(data_dir, 200)
    entry = dataset[100]
    _logger.info(f"{entry['x'].shape=}")
    _logger.info(f"{entry['y'].shape=}")

    dataloader = DataLoader(dataset, batch_size=4, pin_memory=True)
    for entry in dataloader:
        _logger.info(f"{entry.keys()=}")
        _logger.info(f"{entry['x'].shape=}")
        _logger.info(f"{entry['y'].shape=}")
        break
