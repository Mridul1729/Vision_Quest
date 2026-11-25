import os
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

from data_loader import (
    load_ships,
    load_berths,
    load_cranes,
    load_containers,
    load_vehicles,
    load_drivers,
    load_warehouses,
    load_dispatch,
)

app = FastAPI(title="PortSync API", version="1.0")


MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "portsync_model.pkl",
)

model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        model = None


def calculate_port_workload(
    waiting_ships,
    available_berths,
    active_cranes,
    container_count,
):
    ship_pressure = waiting_ships * 10
    berth_pressure = max(0, 5 - available_berths) * 10
    crane_pressure = max(0, 5 - active_cranes) * 10
    container_pressure = min(container_count / 10, 50)

    workload = (
        ship_pressure
        + berth_pressure
        + crane_pressure
        + container_pressure
    )

    return min(round(workload, 2), 100)


def build_port_features():
    ships = load_ships()
    berths = load_berths()
    cranes = load_cranes()
    containers = load_containers()

    waiting_ships = sum(
        1 for ship in ships
        if ship["status"] == "waiting"
    )

    available_berths = sum(
        1 for berth in berths
        if berth["status"] == "available"
    )

    active_cranes = sum(
        1 for crane in cranes
        if crane["status"] == "active"
    )

    container_count = len(containers)

    port_workload = calculate_port_workload(
        waiting_ships,
        available_berths,
        active_cranes,
        container_count,
    )

    return {
        "waiting_ships": waiting_ships,
        "available_berths": available_berths,
        "active_cranes": active_cranes,
        "container_count": container_count,
        "port_workload": port_workload,
    }


def recommend_berth(ship_id):
    ships = load_ships()
    berths = load_berths()

    ship = next(
        (ship for ship in ships if ship["ship_id"] == ship_id),
        None,
    )

    if ship is None:
        return None

    available = [
        berth for berth in berths
        if berth["status"] == "available"
    ]

    if not available:
        return {
            "ship_id": ship_id,
            "recommendation": "No berth available",
        }

    recommended = available[0]

    return {
        "ship_id": ship_id,
        "ship_size": ship["size"],
        "recommended_berth": recommended["berth_id"],
        "reason": "Berth is currently available",
    }


class ShipPredictionInput(BaseModel):
    ship_id: str


@app.get("/")
def root():
    return {"message": "PortSync Backend is running!"}


@app.get("/ships")
def ships():
    return load_ships()


@app.get("/berths")
def berths():
    return load_berths()


@app.get("/cranes")
def cranes():
    return load_cranes()


@app.get("/containers")
def containers():
    return load_containers()


@app.get("/vehicles")
def vehicles():
    return load_vehicles()


@app.get("/drivers")
def drivers():
    return load_drivers()


@app.get("/warehouses")
def warehouses():
    return load_warehouses()


@app.get("/dispatch")
def dispatch():
    return load_dispatch()


@app.get("/features")
def features():
    return build_port_features()


@app.post("/predict")
def predict(data: ShipPredictionInput):
    if model is None:
        return {
            "error": "ML model not found. Put portsync_model.pkl in models/."
        }

    ships = load_ships()
    ship = next(
        (ship for ship in ships if ship["ship_id"] == data.ship_id),
        None,
    )

    if ship is None:
        return {"error": "Ship not found"}

    features = build_port_features()

    input_data = pd.DataFrame([{
        "waiting_ships": features["waiting_ships"],
        "available_berths": features["available_berths"],
        "active_cranes": features["active_cranes"],
        "container_count": features["container_count"],
        "ship_size": ship["size"],
        "port_workload": features["port_workload"],
    }])

    try:
        prediction = model.predict(input_data)[0]
        predicted_wait = max(0, float(prediction))
    except Exception as exc:
        return {"error": f"Prediction failed: {exc}"}

    return {
        "ship_id": ship["ship_id"],
        "ship_name": ship["name"],
        "ship_size": ship["size"],
        "predicted_waiting_time_hours": round(predicted_wait, 2),
        "unit": "hours",
        "features_used": input_data.iloc[0].to_dict(),
    }


@app.get("/recommend-berth/{ship_id}")
def get_berth_recommendation(ship_id: str):
    recommendation = recommend_berth(ship_id)

    if recommendation is None:
        return {"error": "Ship not found"}

    return recommendation
