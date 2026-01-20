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
