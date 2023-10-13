"""Main file for training and regitering the model."""
import argparse
import json
import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utils import load_config


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="Path to config file")
    parser.add_argument("--model_path", type=str, help="model path")
    parser.add_argument(
        "--model_info_output_path", type=str, help="model info output path"
    )
    parser.add_argument("--evaluation_output", type=str, help="evaluation output path")

    return parser.parse_args()


def main(args) -> None:
    """Main function to train and register the model."""
    register_model(args)


def register_model(args) -> None:
    """Function to register the trained model."""
    cfg = load_config(args.cfg)

    deploy_flag = 1

    with open((args.evaluation_output / Path("deploy_flag")), "rb") as infile:
        deploy_flag = int(infile.read())

    mlflow.log_metric("deploy flag", int(deploy_flag))

    if deploy_flag == 1:
        print("Registering ", cfg["model"]["registered_name"])

        # load model
        model = mlflow.sklearn.load_model(args.model_path)

        # log model using mlflow
        mlflow.sklearn.log_model(model, cfg["model"]["registered_name"])

        # register logged model using mlflow
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/{cfg['model']['registered_name']}"
        mlflow_model = mlflow.register_model(model_uri, cfg["model"]["registered_name"])
        model_version = mlflow_model.version

        # write model info
        model_info = {
            "id": "{0}:{1}".format(cfg["model"]["registered_name"], model_version)
        }
        output_path = args.model_info_output_path / Path("model_info.json")

        with open(output_path, "w") as of:
            json.dump(model_info, fp=of)

    else:
        print("Model will not be registered!")


if __name__ == "__main__":
    mlflow.start_run()
    mlflow.sklearn.autolog()

    args = parse_args()

    main(args)

    mlflow.end_run()
