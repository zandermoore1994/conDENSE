# conDENSE

This repository implements a version of the model conDENSE from the paper: "conDENSE Conditional Density Estimation for Time Series Anomaly Detection."

TODO: link to paper

# Setup 

This model has a dependency on Tensorflow Probability and at the time of writing requires python version <3.11.
We recommend using a pyenv virtual environment, the helper script `setup.sh` will create and provision a virtual environment for this project.


# Running conDENSE

A helper script has been created to fascilitate running conDENSE on your data. The command below will run the model on sample data:

    `python run.py --data_path data --univariate False`

Please note that the argument data_path must point towards a directory with the following:
- train.npy: normal time series to be used for training, ideally without anomalies. Shape: (n_train_timesteps, n_features)
- test.npy: test time series to be used for model evaluation, ideally with anomalies. Shape: (n_test_timesteps, n_features)
- labels.npy: binary labels indicating if each timestep in the test set is anomalous. Shape: (n_test_timesteps,)

For optimal performance we recommend a per feature min max scaling of the data (using min/max values extracted from the training set).

N.B. `--univariate` arg is set to False by default but it must be included if training on univariate timeseries.
