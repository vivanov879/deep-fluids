""" Utils for building datasets """
from pathlib import Path
from typing import List

from loguru import logger as _logger


def find_npz_files(dir: Path) -> List[Path]:
    _logger.info(f"Building file index for directory {dir}")
    fns = []
    for fn in dir.glob("*.npz"):
        fns.append(fn)
    return fns



