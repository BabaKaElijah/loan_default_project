# Loan Default Prediction System
[![Pipeline](https://img.shields.io/badge/Pipeline-Verified-2ea44f)](README.md)

End-to-end machine learning project that predicts loan default risk and serves
predictions through both a web UI and a REST API.

## Why This Project
Financial institutions must assess loan risk accurately. This project focuses on
production-ready ML engineering: reproducible training, modular pipelines, and
deployable inference services.

## What It Does
- Predicts default risk (0 = unlikely, 1 = likely)
- Returns probability of default
- Exposes inference via FastAPI and Streamlit

## Tech Stack
- Python, Pandas, NumPy, scikit-learn, Joblib
- FastAPI (REST API), Streamlit (interactive UI)
- Docker, GitHub Actions

## Architecture (High Level)
- Data ingestion: load and split raw loan data
- Data transformation: preprocessing and feature encoding
- Model training: train and persist best classifier pipeline
- Prediction pipeline: load persisted pipeline for inference
- API + UI: FastAPI endpoints and Streamlit app

## Quickstart
### 1) Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Train the model
```bash
python -m src.pipelines.train_pipeline
```

### 4) Run the FastAPI app
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 5000
```
Open:
```bash
http://127.0.0.1:5000
```
Docs:
```bash
http://127.0.0.1:5000/docs
```

### 5) Run the Streamlit app
```bash
streamlit run app_streamlit.py
```

## API Example
```bash
curl -X POST http://127.0.0.1:5000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"Age\":35,\"Income\":500000,\"LoanAmount\":200000,\"CreditScore\":600,\"MonthsEmployed\":60,\"NumCreditLines\":3,\"InterestRate\":12.5,\"LoanTerm\":36,\"DTIRatio\":0.9,\"Education\":\"Bachelor\",\"EmploymentType\":\"Salaried\",\"MaritalStatus\":\"Single\",\"HasMortgage\":\"No\",\"HasDependents\":\"Yes\",\"LoanPurpose\":\"Car\",\"HasCoSigner\":\"No\"}"
```

## Docker
```bash
docker build -t loan-default-app .
docker run -p 5000:5000 loan-default-app
```

## CI Notes
CI validates imports and application startup while skipping pipeline load if
model artifacts are missing.

## Project Structure
```
src/
  components/   # ingestion, transformation, training, prediction
  pipelines/    # training entrypoint
templates/      # HTML UI
static/         # CSS assets
app.py          # FastAPI app
app_streamlit.py
```

## Screenshot
![App Screenshot](LoanDefault.PNG)

## Author
Ellias Sithole
# Loan Default Prediction System
End to End Machine Learning Project

## Overview
This project predicts whether a loan applicant is likely to default on a loan.
It demonstrates a complete machine learning lifecycle from raw data to a production ready application.

The focus is not only on model accuracy but also on engineering practices used in real world ML systems.

The project includes automated training pipelines, model versioning, REST APIs, web interfaces, Docker containerization, and CI using GitHub Actions.

## Problem Statement
Financial institutions must assess loan risk accurately.

Poor predictions result in
- Financial losses from defaults
- Missed revenue from overly strict approvals
This system uses applicant and loan features to predict default risk and provide a probability score.

## Prediction Output
- Prediction = 1
  The loan is likely to default
- Prediction = 0
  The loan is unlikely to default
- Probability
  Model confidence that the loan will default

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit Learn
- Flask
- Streamlit
- Docker
- GitHub Actions
- Joblib

### Project Architecture
The project follows a modular and production oriented structure.
### Data Ingestion
Loads and validates raw loan data.
### Data Transformation
Handles preprocessing, encoding, and feature engineering.
### Model Training
Trains a classification pipeline and saves model artifacts.
### Prediction Pipeline
Loads the trained pipeline and performs inference.
### Flask Application
REST API and web interface for predictions.
### Streamlit Application
Interactive UI for non technical users.
### Docker
Containerized application for consistent deployment.
### CI Pipeline
Automated checks on every push and pull request.

## Key Features
- End to end ML pipeline
- Reusable and modular codebase
- Model artifact versioning
- REST API for predictions
- Streamlit UI for business users
- Dockerized deployment
- CI ready for production workflows
- Graceful handling of missing artifacts in CI

## How to Run Locally
### Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```
### Install dependencies
```bash
pip install -r requirements.txt
```
### Run training pipeline
```bash
python -m src.pipelines.train_pipeline
```
### Run Flask app
```bash
python app.py
```
Open in browser
```bash
http://127.0.0.1:5000
```
### Run Streamlit app
```bash
streamlit run streamlit_app.py
```
## Docker Usage
### Build Docker image
```bash
docker build -t loan-default-app .
```
### Run Docker container
```bash
docker run -p 5000:5000 loan-default-app
```

## CI and GitHub Actions
This project includes a GitHub Actions CI workflow that runs on every push and pull request.

CI checks include
- Dependency installation
- Core module import validation
- Flask app load verification
The workflow is designed to avoid failures caused by missing model artifacts during CI runs.

## App Screanshot
![Picture](LoanDefault.PNG)

## Author
Ellias Sithole
Machine Learning and Data Enthusiast

