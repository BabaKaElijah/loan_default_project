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

