import json, random

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from model import TimeSeriesForecasting


def split_df(
    df: pd.DataFrame, split: str, history_size: int = 120, context_size: int = 30
):
    """
    Create a training / validation samples
    Validation samples are the last context_size rows

    :param df:
    :param split:
    :param history_size:
    :param context_size:
    :return:
    """
    if split == "train":
        end_index = random.randint(history_size, df.shape[0] - context_size)
    elif split in ["val", "test"]:
        end_index = df.shape[0]
    else:
        raise ValueError("Invalid split type, must be 'train', 'val' or 'test' only.")

    label_index = end_index - context_size
    start_index = max(0, label_index - history_size)

    history = df[start_index: label_index]
    targets = df[label_index: end_index]

    return history, targets


def pad_arr(arr: np.ndarray, expected_size: int = 120):
    """
    Pad top of array when there is not enough history
    :param arr:
    :param expected_size:
    :return:
    """
    if arr.size == 0:  # Check if the array is empty
        # Handle the empty array case. This could be by creating an array
        # filled with zeros (or another default value) of the expected size.
        # Here, we assume that the array has a second dimension (features).
        return np.zeros((expected_size, arr.shape[1]))
    else:
        arr = np.pad(arr, [(expected_size - arr.shape[0], 0), (0, 0)], mode="edge")
        return arr


def df_to_np(df: pd.DataFrame, expected_size: int = 120):
    if df.empty:  # Check if the DataFrame is empty
        # Handle the empty DataFrame case. This could be by creating an array
        # filled with zeros (or another default value) of the expected size.
        # Here, we assume that the DataFrame has at least one column.
        return np.zeros((expected_size, df.shape[1]))
    else:
        arr = np.array(df)
        arr = pad_arr(arr, expected_size)
        return arr


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, split, features, target, history_size=120, context_size=30):
        self.dataframe = dataframe
        self.split = split
        self.features = features
        self.target = target
        self.history_size= history_size
        self.context_size = context_size

    def __len__(self):
        return len(self.dataframe) - self.history_size - self.context_size + 1

    def __getitem__(self, idx):
        # Ensure idx is within the range that allows forming a complete sample
        idx = min(idx, len(self.dataframe) - self.history_size - self.context_size)
        # Select a window of the DataFrame that includes enough rows to form both history and target
        df = self.dataframe.iloc[idx: idx + self.history_size + self.context_size]
        # df = self.dataframe.iloc[idx: idx + 1]

        src, trg = split_df(df, split=self.split)

        src = src[self.features + [self.target]]
        src = df_to_np(src)

        trg_in = trg[self.features + [f"{self.target}_lag_1"]]

        trg_in = np.array(trg_in)
        trg_out = np.array(trg[self.target])

        src = torch.tensor(src, dtype=torch.float)
        trg_in = torch.tensor(trg_in, dtype=torch.float)
        trg_out = torch.tensor(trg_out, dtype=torch.float)

        return src, trg_in, trg_out


def train(
    data_csv_path: str,
    feature_target_names_path: str,
    output_json_path: str,
    log_dir: str = "ts_logs",
    model_dir: str = "ts_models",
    batch_size: int = 32,
    epochs: int = 10,
    context_size: int = 30,
):
    data = pd.read_csv(data_csv_path)

    with open(feature_target_names_path) as f:
        feature_target_names = json.load(f)

    data_train = data[~data[feature_target_names["target"]].isna()]

    train_data = Dataset(
        dataframe=data_train,
        split="train",
        features=feature_target_names["features"],
        target=feature_target_names["target"],
    )
    val_data = Dataset(
        dataframe=data_train,
        split="val",
        features=feature_target_names["features"],
        target=feature_target_names["target"],
    )

    print("len(train_data)", len(train_data))
    print("len(val_data)", len(val_data))

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=8,
        shuffle=False,
    )

    model = TimeSeriesForecasting(
        n_encoder_inputs=len(feature_target_names["features"]) + 1,
        n_decoder_inputs=len(feature_target_names["features"]) + 1,
        lr=1e-5,
        dropout=0.1,
    )

    logger = TensorBoardLogger(
        save_dir=log_dir,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        mode="min",
        dirpath=model_dir,
        filename="ts",
    )

    # Check if MPS is supported
    if torch.backends.mps.is_available():
        accelerator = 'mps'
        devices = 1
    elif torch.cuda.is_available():
        accelerator = 'gpu'
        devices = "auto"
    else:
        accelerator = 'cpu'  # fallback to cpu
        devices = 1

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=devices,
        # gpus=1, #in lightning, this is strictly "cuda"
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_loader, val_loader)

    result_val = trainer.test(dataloaders=val_loader)

    output_json = {
        "val_loss": result_val[0]["test_loss"],
        "best_model_path": checkpoint_callback.best_model_path,
    }

    if output_json_path is not None:
        with open(output_json_path, "w") as f:
            json.dump(output_json, f, indent=4)

    return output_json


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv_path")
    parser.add_argument("--feature_target_names_path")
    parser.add_argument("--output_json_path", default=None)
    parser.add_argument("--log_dir")
    parser.add_argument("--model_dir")
    parser.add_argument("--epochs", type=int, default=10)

    args = parser.parse_args()
    # # Ensure model_dir exists
    # os.makedirs(args.model_dir, exist_ok=True)
    # # Ensure log_dir exists
    # os.makedirs(args.log_dir, exist_ok=True)

    train(
        data_csv_path=args.data_csv_path,
        feature_target_names_path=args.feature_target_names_path,
        output_json_path=args.output_json_path,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        epochs=args.epochs,
    )