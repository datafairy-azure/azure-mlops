import json

import requests
import streamlit as st
import yaml

from mlops.utils import ml_connect


cfg = []

with open("..\\config.yml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

ml_client = ml_connect(credential_type="default", cfg=cfg)

endpoint_cred = ml_client.online_endpoints.get_keys(
    name=cfg["deployments"]["endpoint_name"]
).access_token

headers = {
    "Authorization": f"Bearer {endpoint_cred}",
    "Content-type": "application/json",
}

st.title("Calling Azure ML endpoint.")

endpoint = cfg["deployments"]["endpoint_url"]
request_json = []

with open("..\\data\\sample_request.json", "r") as f:
    request_json = json.load(f)

if st.button("Call Endpoint!"):
    res = requests.post(url=endpoint, data=json.dumps(request_json), headers=headers)

    st.subheader(f"status code: {res.status_code}, response: {res.text}")
