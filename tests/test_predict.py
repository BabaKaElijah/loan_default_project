import os
import pytest

from src.components.predict_pipeline import PredictPipeline


def _model_available():
    model_dir = os.path.join("artifacts", "model")
    return os.path.isdir(model_dir) and any(
        name.endswith("_pipeline.pkl") for name in os.listdir(model_dir)
    )


@pytest.mark.skipif(not _model_available(), reason="Model artifacts not available")
def test_predict_returns_types():
    pipeline = PredictPipeline()

    sample_input = {
        "Age": 35,
        "Income": 500000,
        "LoanAmount": 200000,
        "CreditScore": 600,
        "MonthsEmployed": 60,
        "NumCreditLines": 3,
        "InterestRate": 12.5,
        "LoanTerm": 36,
        "DTIRatio": 0.9,
        "Education": "Bachelor",
        "EmploymentType": "Salaried",
        "MaritalStatus": "Single",
        "HasMortgage": "No",
        "HasDependents": "Yes",
        "LoanPurpose": "Car",
        "HasCoSigner": "No"
    }

    result = pipeline.predict(sample_input)

    assert isinstance(result, dict)
    assert "prediction" in result
    assert "probability" in result
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0
