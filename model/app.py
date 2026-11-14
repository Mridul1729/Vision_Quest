import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="PortSync AI Prediction API")

# Load trained models
model = joblib.load("models/portsync_model.pkl")
anomaly_model = joblib.load("models/anomaly_model.pkl")


class PortData(BaseModel):
    waiting_ships: int = Field(ge=0)
    available_berths: int = Field(ge=0)
    active_cranes: int = Field(ge=0)
    container_count: int = Field(ge=0)
    ship_size: str
    port_workload: float = Field(ge=0)


def detect_congestion(data, predicted_wait):

    reasons = []

    waiting_ships = data.waiting_ships
    berths = data.available_berths
    workload = data.port_workload

    # HIGH congestion conditions
    if predicted_wait >= 12:
        reasons.append("High predicted ship waiting time")

    if workload >= 85:
        reasons.append("Very high port workload")

    if waiting_ships >= 9 and berths <= 2:
        reasons.append("Many waiting ships with limited berths")

    if reasons:
        return "HIGH", reasons

    # MEDIUM congestion conditions
    if predicted_wait >= 9:
        reasons.append("Elevated predicted waiting time")

    if workload >= 70:
        reasons.append("Elevated port workload")

    if waiting_ships >= 7 and berths <= 3:
        reasons.append("Increasing vessel queue relative to berth availability")

    if reasons:
        return "MEDIUM", reasons

    return "LOW", ["No major congestion indicators detected"]


@app.get("/")
def home():
    return {"message": "PortSync AI API is running"}


@app.post("/predict")
def predict(data: PortData):

    input_data = pd.DataFrame([data.model_dump()])

    # 1. Waiting time prediction
    prediction = model.predict(input_data)[0]
    predicted_wait = max(0, float(prediction))

    # 2. Anomaly detection
    anomaly_result = anomaly_model.predict(input_data)[0]
    anomaly_score = anomaly_model.score_samples(input_data)[0]

    is_anomaly = anomaly_result == -1

    # 3. Congestion assessment
    congestion_level, reasons = detect_congestion(
        data, predicted_wait
    )

    # 4. Generate alert
    if congestion_level == "HIGH":
        alert = "High congestion detected. Immediate operational review recommended."

    elif congestion_level == "MEDIUM":
        alert = "Moderate congestion risk. Monitor port operations."

    elif is_anomaly:
        alert = "Unusual port activity detected. Further investigation recommended."

    else:
        alert = "Port operations appear normal."

    return {
        "predicted_waiting_time_hours": round(
            predicted_wait, 2
        ),

        "unit": "hours",

        "congestion": {
            "level": congestion_level,
            "alert": alert,
            "reasons": reasons
        },

        "anomaly_detection": {
            "is_anomaly": bool(is_anomaly),
            "status": (
                "Anomaly Detected"
                if is_anomaly else "Normal"
            ),
            "anomaly_score": round(
                float(anomaly_score), 4
            )
        }
    }
