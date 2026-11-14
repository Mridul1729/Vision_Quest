
import os
import pandas as pd
import joblib

from xgboost import XGBRegressor
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("data/port_data.csv")

FEATURES = [
    "waiting_ships",
    "available_berths",
    "active_cranes",
    "container_count",
    "ship_size",
    "port_workload"
]

TARGET = "waiting_time_hours"

X = df[FEATURES]
y = df[TARGET]

numeric_features = [
    "waiting_ships",
    "available_berths",
    "active_cranes",
    "container_count",
    "port_workload"
]

categorical_features = ["ship_size"]


def create_preprocessor():
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"),
         numeric_features),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ])


# -----------------------------
# MODEL 1: XGBOOST PREDICTION
# -----------------------------

model = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    objective="reg:squarederror",
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", create_preprocessor()),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print(f"Mean Absolute Error: {mae:.2f} hours")


# -----------------------------
# MODEL 2: ANOMALY DETECTION
# -----------------------------

anomaly_model = Pipeline([
    ("preprocessor", create_preprocessor()),

    ("detector", IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42
    ))
])

# Learn normal patterns from the available port data
anomaly_model.fit(X)

# Save both models
os.makedirs("models", exist_ok=True)

joblib.dump(pipeline, "models/portsync_model.pkl")
joblib.dump(anomaly_model, "models/anomaly_model.pkl")

print("PortSync prediction model trained and saved!")
print("PortSync anomaly detection model trained and saved!")
