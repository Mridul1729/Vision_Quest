# PortSync AI – ML Backend

## Overview

PortSync AI is the machine learning component of PortSync, designed to support port operations by predicting vessel waiting times and identifying unusual operational patterns.

The backend combines a regression model for waiting-time prediction with an unsupervised anomaly-detection model.

## Features

### 1. Vessel Waiting-Time Prediction

- Uses an **XGBoost Regressor** to estimate vessel waiting time.
- Predicts waiting time in hours.
- Uses port-operational parameters as input.

**Input features:**
- `waiting_ships` – Number of ships waiting.
- `available_berths` – Number of available berths.
- `active_cranes` – Number of active cranes.
- `container_count` – Number of containers.
- `ship_size` – Size category of the vessel.
- `port_workload` – Port workload.

**Prediction output:** Estimated vessel waiting time in hours.

### 2. Anomaly Detection

- Uses **Isolation Forest** to identify unusual combinations of port-operational data.
- Classifies observations as:
  - `Normal`
  - `Anomaly Detected`
- Returns an anomaly score alongside the classification.

The anomaly score indicates how unusual an observation is compared with the patterns learned during training. A lower score indicates a more unusual observation.

**Note:** An anomaly is not automatically a congestion alert. It indicates unusual operational behaviour that may require further investigation.

## Model Training

Training script: `model/train_model.py`

Dataset: `data/port_data.csv`

The training pipeline:

1. Loads the port dataset.
2. Separates input features and the target variable.
3. Handles missing numerical and categorical values.
4. Encodes the categorical `ship_size` feature using One-Hot Encoding.
5. Trains the XGBoost regression model.
6. Evaluates the prediction model using Mean Absolute Error (MAE).
7. Trains the Isolation Forest anomaly-detection model.
8. Saves both trained models.

### Model Configuration

| Component | Algorithm |
|---|---|
| Waiting-time prediction | XGBoost Regressor |
| Anomaly detection | Isolation Forest |
| Numerical preprocessing | Median imputation |
| Categorical preprocessing | Most-frequent imputation + One-Hot Encoding |
| Evaluation metric | Mean Absolute Error (MAE) |

## Model Evaluation

The reported Mean Absolute Error (MAE) from the prediction model was:

**0.73 hours**

MAE measures the average absolute difference between predicted and actual waiting times on the evaluation dataset.

## Backend API

The backend is developed using **FastAPI**.

Application file: `model/app.py`

### Available Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Check API status |
| POST | `/predict` | Predict waiting time and detect anomalies |

### Example Request

```json
{
  "waiting_ships": 8,
  "available_berths": 3,
  "active_cranes": 6,
  "container_count": 3500,
  "ship_size": "large",
  "port_workload": 75.5
}
```

### Example Response

```json
{
  "predicted_waiting_time_hours": 13.49,
  "unit": "hours",
  "anomaly_detection": {
    "status": "Anomaly Detected",
    "anomaly_score": -0.52
  }
}
```

*The response values above are illustrative; actual predictions and anomaly scores depend on the input data and trained models.*

## Project Structure

```text
PortSync/
│
├── data/
│   └── port_data.csv
│
├── model/
│   ├── app.py
│   └── train_model.py
│
├── models/
│   ├── portsync_model.pkl
│   └── anomaly_model.pkl
│
└── AI_README.md
```

## Running the Backend

Run commands from the main PortSync project directory.

### 1. Install dependencies

```bash
pip install fastapi uvicorn pandas joblib xgboost scikit-learn
```

### 2. Train the models

```bash
python model/train_model.py
```

This generates the trained model files inside the `models/` directory.

### 3. Start the API

```bash
uvicorn model.app:app --reload
```

### 4. Test the API

Open:

```text
http://127.0.0.1:8000/docs
```

Use the interactive Swagger interface to test the `/predict` endpoint.

## Future Enhancements

- Integration with real-time port operational data APIs.
- Weather-based waiting-time prediction.
- More advanced congestion detection and alert generation.
- Integration with the PortSync frontend.
- Improved model performance using larger and more representative datasets.

## Technology Stack

- Python
- XGBoost
- Scikit-learn
- Pandas
- FastAPI
- Joblib
- Uvicorn
