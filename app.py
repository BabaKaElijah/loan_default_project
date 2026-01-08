import sys
import os
from flask import Flask, request, jsonify, render_template

from src.logger.logger import get_logger
from src.components.predict_pipeline import PredictPipeline

logger = get_logger(__name__)

app = Flask(__name__)
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


@app.route("/")
def home():
    load_pipeline()  # Ensure pipeline is loaded before serving home page
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    load_pipeline()  # Ensure pipeline is loaded

    if pipeline is None:
        return jsonify({
            "error": "Model pipeline not loaded"
        }), 503

    try:
        # Get input data from JSON or form
        if request.is_json:
            input_data = request.get_json()
        else:
            input_data = request.form.to_dict()
            numeric_fields = [
                "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
                "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio"
            ]
            for field in numeric_fields:
                if field in input_data:
                    input_data[field] = float(input_data[field])

        result = pipeline.predict(input_data)

        return jsonify({
            "prediction": result["prediction"],
            "probability": result["probability"]
        })

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    load_pipeline()  # Load pipeline when running locally
    app.run(host="0.0.0.0", port=5000, debug=True)
