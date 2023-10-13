"""Module to connect to ML workspace."""
from mlflow.tracking import MlflowClient
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azureml.core.authentication import InteractiveLoginAuthentication
import mlflow
import yaml


def load_config(config_path: str) -> dict:
    """Function to load config file."""
    cfg = []
    with open(config_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def mlflow_connect(credential_type: str, cfg: dict):
    """Function to connect to MLflow."""
    ml_client = ml_connect(credential_type, cfg)

    mlflow_tracking_uri = ml_client.workspaces.get(
        ml_client.workspace_name
    ).mlflow_tracking_uri

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    return MlflowClient()


def ml_connect(credential_type: str, cfg: dict) -> MLClient:
    """Function to connect to ML workspace.

    Args:
        cfg: Dict with config values.
        credential_type: Type of credential.

    Returns:
        MLClient.
    """
    credential = get_credential(credential_type)

    try:
        credential.get_token("https://management.azure.com/.default")
    except Exception:
        credential = InteractiveLoginAuthentication(
            tenant_id=cfg["connections"]["tenant_id"]
        )

    return MLClient(
        credential,
        cfg["connections"]["subscription_id"],
        cfg["connections"]["resource_group"],
        cfg["connections"]["workspace"],
    )


def get_credential(credential_type: str) -> DefaultAzureCredential:
    """Function to get credential.

    Args:
        credential_type: Type of credential.

    Returns:
        Credential.
    """
    if credential_type == "default":
        return DefaultAzureCredential()
    elif credential_type == "interactive":
        return InteractiveBrowserCredential()
    else:
        raise ValueError("Invalid credential type.")
