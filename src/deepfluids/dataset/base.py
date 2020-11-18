""" A base dataset class definition """

from pathlib import Path
from typing import List, Dict

from torch.utils.data import DataLoader, Dataset
import numpy as np
from loguru import logger as _logger

from .utils import find_npz_files

class BaseDataset(Dataset):
    def __init__(self, data_dir: Path):
        """
        A base class for datasets
        Args:
            data_dir: data directory
        """
        super().__init__()
        self.fns = find_npz_files(data_dir)

    def __len__(self) -> int:
        return len(self.fns)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        raise NotImplemented

