"""Runner script to train and evaluate conDENSE on processed dataset."""
import numpy as np
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from conDENSE import utils
from conDENSE.dtypes import DatasetLoader
from conDENSE.models.conDENSE import conDENSE
from conDENSE.parser import args


np.random.seed(0)
tf.random.set_seed(0)

if __name__ == "__main__":
    train, test, labels = utils.open_dataset(args.data_path)
    loader = DatasetLoader(
        train=train, test=test, labels=labels, window_size=args.window_size, univariate=args.univariate
    )
    model = conDENSE(n_features=train.shape[1], window_size=args.window_size)
    model.compile(**model.compile_kwargs)
    model.fit(
        loader.fetch_train(),
        loader.fetch_train(),
        epochs=args.n_epochs,
        batch_size=args.batch_size,
        validation_data=(loader.fetch_val(), loader.fetch_val()),
        callbacks=[
            EarlyStopping(
                patience=3,
                verbose=1,
                min_delta=0.001,
                monitor="val_loss",
                mode="auto",
                restore_best_weights=True,
            )
        ],
        verbose=True,
        shuffle=False,
    )

    preds = model.predict(loader.fetch_test())
    preds = utils.clip_inf_values(preds)
    precision, recall, _ = metrics.precision_recall_curve(labels, preds)
    print("ROC AUC:", metrics.roc_auc_score(labels, preds))
    print("PR AUC:", metrics.auc(recall, precision))

