import sys
from flask import Flask, request, jsonify, render_template

from src.logger.logger import get_logger
from src.components.predict_pipeline import PredictPipeline

logger = get_logger(__name__)

app = Flask(__name__)
pipeline = PredictPipeline()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.is_json:
            input_data = request.get_json()
        else:
            input_data = request.form.to_dict()

            numeric_fields = [
                "Age",
                "Income",
                "LoanAmount",
                "CreditScore",
                "MonthsEmployed",
                "NumCreditLines",
                "InterestRate",
                "LoanTerm",
                "DTIRatio"
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
        logger.error(str(e))
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
