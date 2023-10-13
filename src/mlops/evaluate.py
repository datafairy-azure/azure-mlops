"""Main file for training and regitering the model."""
import argparse
import os
from pathlib import Path
import json
import yaml

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import classification_report, r2_score
from mlflow.tracking import MlflowClient
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="path to input data")
    parser.add_argument("--output_folder", type=str, help="path to output folder")
    parser.add_argument("--model_path", type=str, help="Path to the trained model.")
    parser.add_argument("--cfg", type=str, help="Path to the config file.")
    parser.add_argument(
        "--runner", type=str, help="Local or Cloud Runner", default="CloudRunner"
    )
    return parser.parse_args()


def main(args) -> None:
    """Function to score a sklearn model."""
    cfg = []

    with open(args.cfg, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    test_df = pd.read_parquet(Path(args.data_path) / "test.parquet")

    sk_model = mlflow.sklearn.load_model(args.model_path)

    x_test = test_df.iloc[:, :-1]
    y_test = test_df["default payment next month"]

    yhat_test = sk_model.predict(x_test)

    score = r2_score(y_test, yhat_test)

    with open(
        os.path.join(args.output_folder, "classification_report.json"), "x"
    ) as output:
        output.write(
            json.dumps(classification_report(y_test, yhat_test, output_dict=True))
        )
    # ----------------- Model Promotion ---------------- #
    if args.runner == "CloudRunner":
        model_promotion(
            args.output_folder,
            x_test,
            y_test,
            yhat_test,
            score,
            cfg,
        )


def model_promotion(
    evaluation_output, x_test, y_test, yhat_test, score, cfg: dict
) -> tuple:
    """Function to promote a model to a production stage."""
    scores = {}
    predictions = {}
    model_name = cfg["model"]["registered_name"]

    ml_client = MLClient(
        DefaultAzureCredential(),
        cfg["connections"]["subscription_id"],
        cfg["connections"]["resource_group"],
        cfg["connections"]["workspace"],
    )

    mlflow_tracking_uri = ml_client.workspaces.get(
        ml_client.workspace_name
    ).mlflow_tracking_uri
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    client = MlflowClient()

    versions = [
        model_run.version
        for model_run in client.search_model_versions(f"name='{model_name}'")
    ]

    for model_version in versions[0:2]:
        mdl = mlflow.sklearn.load_model(
            model_uri=f"models:/{model_name}/{model_version}"
        )
        predictions[f"{model_name}:{model_version}"] = mdl.predict(x_test)
        scores[f"{model_name}:{model_version}"] = r2_score(
            y_test, predictions[f"{model_name}:{model_version}"]
        )

    if scores:
        if score >= max(list(scores.values())):
            deploy_flag = 1
        else:
            deploy_flag = 0
    else:
        deploy_flag = 1
    print(f"Deploy flag: {deploy_flag}")

    with open((Path(evaluation_output) / "deploy_flag"), "w") as outfile:
        outfile.write(f"{int(deploy_flag)}")

    # add current model score and predictions
    scores["current model"] = score
    predictions["currrent model"] = yhat_test

    perf_comparison_plot = pd.DataFrame(scores, index=["r2 score"]).plot(
        kind="bar", figsize=(15, 10)
    )
    perf_comparison_plot.figure.savefig(cfg["model"]["perf_comparison_file_name"])
    perf_comparison_plot.figure.savefig(
        Path(evaluation_output) / cfg["model"]["perf_comparison_file_name"]
    )

    mlflow.log_metric("deploy flag", bool(deploy_flag))
    mlflow.log_artifact(cfg["model"]["perf_comparison_file_name"])

    return predictions, deploy_flag


if __name__ == "__main__":
    mlflow.start_run()
    mlflow.sklearn.autolog()

    args = parse_args()

    main(args)

    mlflow.end_run()
