"""Module for creating an environment in Azure ML."""
from azure.ai.ml.entities import Environment

from mlops.ml_connect import ml_connect

import yaml

cfg = []

with open("config/config.yml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

ml_client = ml_connect(credential_type="default", cfg=cfg)

env = Environment(
    image=["compute"]["image"],
    conda_file=["compute"]["conda_file"],
    name=cfg["compute"]["environment_name"],
    description="Environment created from a Docker image plus Conda environment.",
)
ml_client.environments.create_or_update(env)
