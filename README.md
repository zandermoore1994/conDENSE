# conDENSE: Conditional Density Estimation for Time Series Anomaly Detection

An implementation of the conDENSE model for time series anomaly detection, based on the research paper published in the Journal of Artificial Intelligence Research.

## Paper Reference

For more details, see the original paper: [conDENSE: Conditional Density Estimation for Time Series Anomaly Detection](https://www.jair.org/index.php/jair/article/view/14849)

```bibtex
@article{moore2024condense,
  title={conDENSE: Conditional Density Estimation for Time Series Anomaly Detection},
  author={Moore, Alex and Morelli, Davide},
  journal={Journal of Artificial Intelligence Research},
  volume={79},
  pages={801--824},
  year={2024}
}
```

## Prerequisites

- Python version: >=3.10, <3.12 
- Package manager: Poetry

We recommend using `pyenv` for Python version management.

## Installation

1. Install Poetry:

   ```bash
   pip install poetry
   ```

2. Install project dependencies:
   ```bash
   poetry install
   ```

## Usage

Run conDENSE using the provided helper script:

```bash
poetry run python run.py --data_path data
```

### Data Requirements

The `data_path` directory must contain the following NumPy files:

| File | Description | Shape |
|------|-------------|-------|
| `train.npy` | Training data (normal time series without anomalies) | `(n_train_timesteps, n_features)` |
| `test.npy` | Test data (time series with anomalies) | `(n_test_timesteps, n_features)` |
| `labels.npy` | Binary anomaly labels for test set | `(n_test_timesteps,)` |

### Data Preprocessing

For optimal performance:
- Apply per-feature min-max scaling to your data
- Use scaling parameters derived from the training set
- Set `--univariate True` when working with univariate time series (default is False)