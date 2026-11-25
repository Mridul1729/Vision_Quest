"""
Training helper for the PortSync waiting-time model.

The original project model can be kept here. This script is only a
reference/training entry point. It expects data/port_data.csv with:
waiting_ships, available_berths, active_cranes, container_count,
ship_size, port_workload, waiting_time_hours
"""

import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT, "data", "port_data.csv")
MODEL_PATH = os.path.join(ROOT, "models", "portsync_model.pkl")


def train():
    df = pd.read_csv(DATA_PATH)

    features = [
        "waiting_ships",
        "available_berths",
        "active_cranes",
        "container_count",
        "ship_size",
        "port_workload",
    ]
    target = "waiting_time_hours"

    X = df[features]
    y = df[target]

    numeric_features = [
        "waiting_ships",
        "available_berths",
        "active_cranes",
        "container_count",
        "port_workload",
    ]
    categorical_features = ["ship_size"]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                random_state=42,
            )),
        ]
    )

    pipeline.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()
