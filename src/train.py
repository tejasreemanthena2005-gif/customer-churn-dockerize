import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# Load dataset
df = pd.read_csv("data/customer_churn.csv")


# Features and target
target = "Churn"

features = [
    c for c in df.columns
    if c not in ["CustomerID", target]
]

X = df[features]

y = df[target].map({
    "No": 0,
    "Yes": 1
})

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ML pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

# Train
pipeline.fit(X_train, y_train)

# Prediction
y_pred = pipeline.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Save model
joblib.dump(
    pipeline,
    "models/churn_model.pkl"
)

print("Model saved successfully!")