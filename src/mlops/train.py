"""Main file for training and regitering the model."""
import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", required=False, default=100, type=int)
    parser.add_argument("--learning_rate", required=False, default=0.1, type=float)
    parser.add_argument("--model_output", type=str, help="Path of output model")
    parser.add_argument("--data_path", type=str, help="Path to datasets")

    return parser.parse_args()


def main(args) -> None:
    """Main function to train and register the model."""
    train_df = pd.read_parquet(Path(args.data_path) / "train.parquet")
    test_df = pd.read_parquet(Path(args.data_path) / "test.parquet")

    y_train = train_df.pop("default payment next month")
    x_train = train_df.values
    y_test = test_df.pop("default payment next month")
    x_test = test_df.values

    clf = train_model(args.learning_rate, args.n_estimators, x_train, y_train)

    y_pred = clf.predict(x_test)

    print(classification_report(y_test, y_pred))

    save_model_to_file(args.model_output, clf)


def train_model(
    learning_rate: float,
    n_estimators: int,
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
) -> GradientBoostingClassifier:
    """Function to read and split the data in train and test sets.

    Args:
        learning_rate: Learning rate.
        n_estimators: Number estimators.
        x_train: Train data set dataframe.
        y_train: Test label set dataframe.

    Returns:
        Gradient boosting classifier.
    """
    clf = GradientBoostingClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate
    )
    clf.fit(x_train, y_train)

    return clf


def save_model_to_file(model_output: str, clf: GradientBoostingClassifier) -> None:
    """Function to save the model to file.

    Args:
        model_output: Path for saving the model.
        clf: Gradient boosting classifier.
    """
    mlflow.sklearn.save_model(sk_model=clf, path=model_output)


if __name__ == "__main__":
    mlflow.start_run()
    mlflow.sklearn.autolog()

    args = parse_args()

    main(args)

    mlflow.end_run()
