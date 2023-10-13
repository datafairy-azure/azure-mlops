"""Main file for training and regitering the model."""
import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

import os, sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utils import load_config


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", type=str, help="Path to raw data")
    parser.add_argument("--test_train_ratio", type=float, required=False, default=0.75)
    parser.add_argument("--cfg", type=str, help="Path to config file")
    parser.add_argument("--data_path", type=str, help="Path to all files")
    parser.add_argument("--enable_monitoring", type=str, help="enable logging to ADX")
    parser.add_argument(
        "--table_name",
        type=str,
        default="mlmonitoring",
        help="Table name in ADX for logging",
    )

    return parser.parse_args()


# TODO: Add logging to ADX
# def log_training_data(df, table_name):
#     """Log training data to ADX"""
#     from obs.collector import Online_Collector
#     collector = Online_Collector(table_name)
#     collector.batch_collect(df)


def main(args, cfg) -> None:
    """Main function to train and register the model."""

    df = load_data(args.raw_data)

    train_df, test_df = prepare_data(data=df, test_train_ratio=args.test_train_ratio)

    mlflow.log_metric("train size", train_df.shape[0])
    mlflow.log_metric("test size", test_df.shape[0])

    train_df.to_parquet(args.data_path / (Path(cfg["data"]["train_data"])))
    test_df.to_parquet(args.data_path / (Path(cfg["data"]["test_data"])))

    # TODO: Add logging to ADX
    # if (args.enable_monitoring.lower() == 'true' or args.enable_monitoring == '1' or args.enable_monitoring.lower() == 'yes'):
    #     log_training_data(data, args.table_name)


def load_data(path: str) -> pd.DataFrame:
    """Function to load the data from the source.

    Args:
        path: File path.

    Returns:
        Dataframe.
    """
    return pd.read_csv(path, header=1, index_col=0)


def prepare_data(data: pd.DataFrame, test_train_ratio: float) -> pd.DataFrame:
    """Function to read and split the data in train and test sets.

    Args:
        test_train_ratio: Test-train ratio.
        data: Dataframe.

    Returns:
        Train and test split sets.
    """
    mlflow.log_metric("num_samples", data.shape[0])
    mlflow.log_metric("num_features", data.shape[1] - 1)

    msk = np.random.rand(len(data)) < test_train_ratio

    train_df = data[msk]
    test_df = data[~msk]

    return train_df, test_df


if __name__ == "__main__":
    mlflow.start_run()
    mlflow.sklearn.autolog()

    args = parse_args()

    cfg = load_config(args.cfg)

    lines = [
        f"Raw data path: {args.raw_data}",
        f"Train dataset output path: {cfg['data']['train_data']}",
        f"Test dataset path: {cfg['data']['test_data']}",
    ]

    for line in lines:
        print(line)

    main(args, cfg)

    mlflow.end_run()
