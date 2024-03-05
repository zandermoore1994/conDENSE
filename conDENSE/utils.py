"""Utilities module."""
import os
from typing import Optional, Tuple

import numpy as np


def open_dataset(dataset: str = "data") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Open a preprocessed dataset.

    Dataset must be a named dir split into numpy files named train.npy, test.npy,
    labels.npy.

    Args:
        dataset (str): name of dir holding dataset (relative to root dir)
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: loaded train, test and labels
        arrays.
    """
    current_dir = os.path.dirname(__file__)
    rel_path = f"../{dataset}/"
    return (
        np.load(os.path.join(current_dir, rel_path + "train.npy")),
        np.load(os.path.join(current_dir, rel_path + "test.npy")),
        np.load(os.path.join(current_dir, rel_path + "labels.npy")),
    )


def clip_inf_values(array: np.ndarray) -> np.ndarray:
    """Clip infinite values in an array to max observed finite value.

    Args:
        array (np.ndarray): array to clip
    Returns:
        np.ndarray: clipped array.
    """
    array = np.where(np.isinf(array), array[np.invert(np.isinf(array))].max(), array)
    array = np.where(np.isnan(array), np.nanmax(array), array)
    return array