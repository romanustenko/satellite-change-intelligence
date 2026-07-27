"""Dataset loading, preprocessing and augmentation utilities."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import gdown




def download_levir_cd(
    download_url: str,
    download_directory: str | Path,
) -> Path:
    """Download the LEVIR-CD dataset into the specified directory."""

    download_directory = Path(download_directory)

    download_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    download_files = gdown.download_folder(
        url=download_url,
        output=str(download_directory),
        quiet=False,
    )

    if not download_files:
        raise RuntimeError("LEVIR-CD download failed.")

    return download_directory