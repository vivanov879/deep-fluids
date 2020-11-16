""" Autoencoder dataset """

from pathlib import Path
from typing import Dict

import numpy as np
from loguru import logger as _logger
from torch.utils.data import DataLoader

from ...dataset.base import BaseDataset


class AutoencoderDataset(BaseDataset):
    def __init__(self, data_dir: Path):
        super().__init__(data_dir)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        fn = self.fns[idx]
        fn = str(fn)
        data = np.load(fn)
        data = {'x': np.array(data['x'].transpose(3, 0, 1, 2), dtype=np.float32),  # CDHW
                'y': np.array(data['y'][:, -1], dtype=np.float32)}
        return data


if __name__ == '__main__':
    data_dir = Path("/Users/vivanov/Projects/deep-fluids/data/smoke3_mov200_f400/v")
    dataset = AutoencoderDataset(data_dir)
    entry = dataset[100]
    _logger.info(f"{entry['x'].shape=}")
    _logger.info(f"{entry['y'].shape=}")

    dataloader = DataLoader(dataset, batch_size=4, pin_memory=True)
    for entry in dataloader:
        _logger.info(f"{entry.keys()=}")
        _logger.info(f"{entry['x'].shape=}")
        _logger.info(f"{entry['y'].shape=}")
        break
