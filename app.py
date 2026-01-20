import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from src.logger.logger import get_logger
from src.components.predict_pipeline import PredictPipeline

logger = get_logger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory="templates")
pipeline = None  # Will be loaded lazily


def load_pipeline():
    """Load model pipeline unless CI skips it."""
    global pipeline
    if pipeline is not None:
        return

    if os.getenv("SKIP_PIPELINE_LOAD") == "true":
        logger.info("Skipping pipeline load (CI mode)")
        return

    try:
        pipeline = PredictPipeline()
        logger.info("Model pipeline loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model pipeline: {str(e)}")


@app.get("/")
def home(request: Request):
    load_pipeline()  # Ensure pipeline is loaded before serving home page
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(request: Request):
    load_pipeline()  # Ensure pipeline is loaded

    if pipeline is None:
        return JSONResponse(
            {"error": "Model pipeline not loaded"},
            status_code=503
        )

    try:
        input_data = {}
        if request.headers.get("content-type", "").startswith("application/json"):
            input_data = await request.json()
        else:
            form_data = await request.form()
            input_data = dict(form_data)
            numeric_fields = [
                "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
                "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio"
            ]
            for field in numeric_fields:
                if field in input_data:
                    input_data[field] = float(input_data[field])

        result = pipeline.predict(input_data)

        return JSONResponse({
            "prediction": result["prediction"],
            "probability": result["probability"]
        })

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)
