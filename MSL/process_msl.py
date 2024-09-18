"""Script to process msl dataset."""
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


INSTANCES = ["P-1", "T-1", "A-4", "C-2"]
if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    anoms = pd.read_csv(os.path.join(current_dir, "raw/labeled_anomalies.csv"))
    for ins in INSTANCES:
        for set in ["train", "test"]:
            data = np.load(os.path.join(current_dir, f"raw/{set}/{ins}.npy"))
            np.save(os.path.join(current_dir, f"{ins}_{set}.npy"), data)
            if set == "test":
                labels = np.zeros(data.shape[0])
                anom_inds = anoms.loc[
                    anoms["chan_id"] == ins, "anomaly_sequences"
                ].values[0]
                anom_inds = anom_inds.replace("]", "").replace("[", "").split(", ")
                anom_inds = [int(i) for i in anom_inds]
                for i in range(0, len(anom_inds), 2):
                    labels[anom_inds[i] : anom_inds[i + 1]] = 1
                np.save(os.path.join(current_dir, f"{ins}_labels.npy"), labels)
