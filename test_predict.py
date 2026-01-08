from src.components.predict_pipeline import PredictPipeline

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
print(result)
