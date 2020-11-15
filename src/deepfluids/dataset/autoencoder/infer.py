""" Autoencoder dataset """
import os
from pathlib import Path
from typing import Dict

import numpy as np
from loguru import logger as _logger
from torch.utils.data import DataLoader

from ...dataset.base import BaseDataset


class AutoencoderInferenceDataset(BaseDataset):
    def __init__(self, data_dir: Path, num_frames: int):
        super().__init__(data_dir)
        self.num_frames = num_frames
        self.fns = list(sorted(self.fns, key=self._sortf))

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        fn = self.fns[idx]
        fn = str(fn)
        data = np.load(fn)
        data = {'x': np.asarray(data['x'].transpose(3, 0, 1, 2), dtype=np.float32),  # CDHW
                'y': np.asarray(data['y'][:, -1], dtype=np.float32)}
        return data

    def _sortf(self, x):
        n = os.path.basename(x)[:-4].split('_')
        return int(n[0]) * self.num_frames + int(n[1])

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
