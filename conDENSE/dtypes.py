"""Dtypes module for conDENSE."""
from typing import List, Optional, Union

import numpy as np


class DatasetLoader:
    """Class to store and load processed datasets."""

    def __init__(
        self,
        train: np.ndarray,
        test: np.ndarray,
        labels: np.ndarray,
        window_size: Optional[int] = 20,
        univariate: bool = False,
        validation_size: float = 0.2,
    ):
        """Initialise DatasetLoader class.

        Args:
            train (np.ndarray): training data, shape=(timesteps, features)
            test (np.ndarray): test data, shape=(timesteps, features)
            labels (np.ndarray): labels for test data, shape=(timesteps)
            window_size (Optional[int]): window size for input into VAE, if None no conditional
                input produced (usef for unconditional MAF).
            univariate (bool): flag indicating if timeseries is univariate
            validation_size (float): portion of training set to use for validation.
        """
        self.select_validation_indicies(train, validation_size)
        if window_size:
            self.test_vae = self.apply_sliding_window(test, window_size, univariate)
            train_vae = self.apply_sliding_window(train, window_size, univariate)
            self.train_vae = train_vae[self.train_inds, :, :]
            self.val_vae = train_vae[self.val_inds, :, :]
        self.train_maf = train[self.train_inds, :]
        self.val_maf = train[self.val_inds, :]
        self.test_maf = test
        self.labels = labels

    def select_validation_indicies(self, train: np.ndarray, validation_size: float):
        """Select indicies for training and validation sets.

        Args:
            train (np.ndarray): raw training data
            validation_size (float): portion of the training set to be used for validation.
        """
        n = len(train)
        self.val_inds = np.random.choice(
            n, size=int(n * validation_size), replace=False
        )
        self.train_inds = [i for i in range(n) if ~np.isin(i, self.val_inds)]
    

    def apply_sliding_window(
        self, data: np.ndarray, window_size: int, univariate: bool = True
    ) -> np.ndarray:
        """Apply sliding window to time series.

        Args:
            data (np.ndarray): original time series, shape=(timesteps, features)
            w_size (int): window size
            univariate (bool): flag indicating if time series is univariate
        Returns:
            np.ndarray: processed data, shape=(timesteps, w_size, features)

        N.B. if univariate output array will be 2D.
        """
        if univariate:
            data = np.squeeze(data)
            data = np.concatenate([data[0].repeat(window_size - 1), data])
            return np.lib.stride_tricks.sliding_window_view(
                data, window_shape=window_size, axis=0
            )[..., np.newaxis]
        else:
            length = len(data)
            data = np.squeeze(data)
            data = np.concatenate([data[:1, :].repeat(window_size - 1, axis=0), data])
            return np.lib.stride_tricks.sliding_window_view(
                data, window_shape=window_size, axis=0
            ).reshape((length, window_size, -1))

    def fetch_train(self) -> Union[List[np.ndarray], np.ndarray]:
        """Fetch training data.

        Returns:
            Union[List[np.ndarray], np.ndarray]: training data, if window_size
                passed to init then this will return a list containing maf and
                vae inputs.
        """
        if hasattr(self, "train_vae"):
            return [self.train_maf, self.train_vae]
        else:
            return self.train_maf

    def fetch_val(self) -> Union[List[np.ndarray], np.ndarray]:
        """Fetch validation data.

        Returns:
            Union[List[np.ndarray], np.ndarray]: validation data, if window_size
                passed to init then this will return a list containing maf and
                vae inputs.
        """
        if hasattr(self, "val_vae"):
            return [self.val_maf, self.val_vae]
        else:
            return self.val_maf

    def fetch_test(self) -> Union[List[np.ndarray], np.ndarray]:
        """Fetch test data.

        Returns:
            Union[List[np.ndarray], np.ndarray]: test data, if window_size
                passed to init then this will return a list containing maf and
                vae inputs.
        """
        if hasattr(self, "test_vae"):
            return [self.test_maf, self.test_vae]
        else:
            return self.test_maf
