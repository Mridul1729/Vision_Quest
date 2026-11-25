# PortSync

PortSync is a smart port-operations platform prototype for coordinating ships,
berths, cranes, trucks, drivers, containers, warehouses and dispatch.

## Project structure

```text
PortSync_Project/
├── api.py
├── data_loader.py
├── data/
│   ├── ships.json
│   ├── berths.json
│   ├── cranes.json
│   ├── containers.json
│   ├── vehicles.json
│   ├── drivers.json
│   ├── warehouses.json
│   └── dispatch.json
├── model/
│   └── train_model.py
├── models/
│   └── portsync_model.pkl
└── README.md
```

## What is already built

- JSON-based operational data layer
- Python data loader
- FastAPI backend
- Port-wide feature calculation
- Port workload calculation
- ML waiting-time prediction endpoint
- Ship-specific prediction
- Basic berth recommendation
- Vehicle, driver, warehouse and dispatch data endpoints

## Main operational flow

Ship arrival
→ berth allocation
→ crane assignment
→ truck/vehicle allocation
→ driver assignment
→ container movement
→ warehouse/yard
→ road/rail dispatch

## Run the backend

Install dependencies:

```bash
pip install fastapi uvicorn pandas joblib scikit-learn xgboost
```

Start the API:

```bash
uvicorn api:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

Swagger API testing:

```text
http://127.0.0.1:8000/docs
```

## Useful endpoints

GET `/`
GET `/ships`
GET `/berths`
GET `/cranes`
GET `/containers`
GET `/vehicles`
GET `/drivers`
GET `/warehouses`
GET `/dispatch`
GET `/features`

GET `/recommend-berth/{ship_id}`

POST `/predict`

For `/predict`, use JSON like:

```json
{
  "ship_id": "SHIP001"
}
```

## ML model

The existing model expects these six inputs:

- waiting_ships
- available_berths
- active_cranes
- container_count
- ship_size
- port_workload

The backend builds the first five port/ship values from the JSON data and sends
them to the trained model. The training CSV should not be replaced just because
new operational dummy data is added.

## What remains for the full PortSync demo

1. Full allocation engine for berth, crane, vehicle and driver.
2. Container lifecycle tracking.
3. Warehouse and yard capacity logic.
4. Rail/road dispatch recommendations.
5. Dashboard frontend.
6. Live port map.
7. Digital-twin style visualization.
8. What-if disruption simulation.
9. Alerts and KPI analytics.
10. Final integration and demo flow.

## Development rule

Keep the data layer, ML layer, allocation logic, API and frontend separate.

The ML model is one part of PortSync, not the entire PortSync system.
