import joblib
import pandas as pd


# Load trained model
model = joblib.load("models/churn_model.pkl")
print("Customer Churn Model Loaded Successfully")


# Sample customer
customer = pd.DataFrame([
    {
        "Age": 45,
        "Tenure": 5,
        "MonthlyBill": 3500,
        "SupportCalls": 6
    }
])


# Prediction
prediction = model.predict(customer)

# Probability
probability = model.predict_proba(customer)


if prediction[0] == 1:
    result = "Customer is likely to CHURN"
else:
    result = "Customer is likely to STAY"


print("--------------------------------")
print("Prediction:", result)
print("Churn Probability:", probability[0][1])
print("--------------------------------")