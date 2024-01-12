# Getting started
- Create a virtual environment: python -m venv .venv
- Activate the venv: source .venv/bin/activate
- pip install --upgrade setuptools
- pip install --upgrade pip
- pip install -r requirements.txt
- pip install src to install the mlops package
- fill out config.yml and move it to ./config
- setup pre-comit hooks: pre-commit install

# Understand the code base
- ./notebooks: Jupyter notebooks for experiments
- ./models: Packaged machine learning models
- ./pipelines: DevOps pipelines
- ./jobs: Deployment jobs
- ./src: Model training code
- [Github: Azure/azureml-examples](https://github.com/Azure/azureml-examples)
- [Github: MSLearn AzureML] https://github.com/MicrosoftLearning/mslearn-azure-ml
- https://github.com/Azure/mlops-v2-gha-demo/

# MLOps workflow
0. Notebook with promising model
1. Clean the code, add linting and testing and create a code base (package)
2. Prep the data for model training
3. Train the model with MLflow
4. Score the model
5. Register the model with MLflow
6. Create an endpoint and deploy the model
7. Validate the model using the endpoint
8. Integrate the endpoint with a web app
9. Monitor
10. Retrain

# How to create a MLOps workflow
## The training pipeline
### Clean the code base
- add the experimental notebook to ./notebooks
- move the code from the notebook to ./src directory and create a main.py file
- create functions, clean up the code, write unit tests and add liniting
- pytest and flake8 should pass when running in the ./src directory

### Create a script to prep the data
- The data needs to be splint into train/validate/test data sets
- The data needs to be uploaded to the blob

### Create a script to train the model
- In ./src update the file train.py
- In pipelines/model-training.yml update the parameters for train.py

### Evaluate the model
- In ./src update the file evaluate.py
- In pipelines/model-training.yml update the parameters for evaluate.py

### Register the model
- In ./src update the file register.py
- In pipelines/model-training.yml update the parameters for register.py

## The deployment pipeline
### Create or update the endpoint
- Once the predictions for the test-set are accepted update the endpoint with the newly trained model
- In ./jobs/yaml run create_or_update_endpoint.yml
- In ./jobs/yaml run deploy_model_to_endpoint.yml
- In ./jobs/yaml run score_endpoint.yml

 ### Deploy to production
 - After all the above steps have passed the endpoint in production can be updated

 ### Monitor a model
 - https://learn.microsoft.com/en-us/azure/machine-learning/how-to-collect-production-data?view=azureml-api-2&tabs=python

 # Extras
 ## Test coverage
Assuming you have file calles sample.py and test written in test.py. Use the following code to run tests with coverage:

```py.test test.py --cov=sample.py```

To get the coverage report use:

```coverage report -m```
