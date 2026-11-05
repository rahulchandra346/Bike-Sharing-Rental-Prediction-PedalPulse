
import joblib
import logging
import pandas as pd
import numpy as np

from pathlib import Path
from datetime import date
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

# ==========================================
# 1. LOAD OR TRAIN MODEL
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH_JOBLIB = BASE_DIR / "bike_rental_model.joblib"
MODEL_PATH_PICKLE = BASE_DIR / "bike_rental_model.pkl"
DATASET_PATH = BASE_DIR / "Dataset.csv"
FEATURE_COLUMNS = [
    "season", "yr", "holiday", "workingday", "weathersit", "temp", "atemp",
    "hum", "windspeed", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "month_sin", "month_cos", "day_of_month", "day_of_year"
]
CATEGORICAL_FEATURES = ["season", "holiday", "workingday", "weathersit"]


def build_training_features(data):
    dates = data["dteday"]
    weekday = (dates.dt.weekday + 1) % 7
    month = dates.dt.month
    hour = data["hr"]

    return pd.DataFrame({
        "season": data["season"],
        "yr": dates.dt.year,
        "holiday": data["holiday"],
        "workingday": data["workingday"],
        "weathersit": data["weathersit"],
        "temp": data["temp"],
        "atemp": data["atemp"],
        "hum": data["hum"],
        "windspeed": data["windspeed"],
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "weekday_sin": np.sin(2 * np.pi * weekday / 7),
        "weekday_cos": np.cos(2 * np.pi * weekday / 7),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "day_of_month": dates.dt.day,
        "day_of_year": dates.dt.dayofyear
    }, columns=FEATURE_COLUMNS)


def train_dataset_model(dataset_path):
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"No usable model artifact was found and training data is missing: {dataset_path}"
        )

    data = pd.read_csv(dataset_path, na_values=["?"])
    required_columns = {
        "dteday", "hr", "season", "holiday", "workingday", "weathersit",
        "temp", "atemp", "hum", "windspeed", "cnt"
    }
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"Cannot train fallback model; dataset is missing columns: {', '.join(sorted(missing_columns))}"
        )

    data["dteday"] = pd.to_datetime(
        data["dteday"], format="%d-%m-%Y", errors="coerce"
    )
    numeric_columns = ["hr", "temp", "atemp", "hum", "windspeed", "cnt"]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=list(required_columns)).copy()
    if data.empty:
        raise ValueError("Cannot train fallback model; the dataset contains no valid training rows.")

    features = build_training_features(data)
    numeric_features = [
        column for column in FEATURE_COLUMNS
        if column not in CATEGORICAL_FEATURES
    ]
    preprocessing = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("numeric", "passthrough", numeric_features)
    ])
    trained_model = make_pipeline(
        preprocessing,
        ExtraTreesRegressor(
            n_estimators=100,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    )
    trained_model.fit(features, data["cnt"])
    logging.warning(
        "Using a model trained from %s because no usable model artifact was found.",
        dataset_path.name
    )
    return (
        trained_model,
        "Dataset-trained Extra Trees fallback",
        FEATURE_COLUMNS,
        data["dteday"].min().date(),
        data["dteday"].max().date()
    )

model = None
if MODEL_PATH_JOBLIB.is_file():
    artifact = joblib.load(MODEL_PATH_JOBLIB)
elif MODEL_PATH_PICKLE.is_file():
    with MODEL_PATH_PICKLE.open("rb") as model_file:
        is_lfs_pointer = model_file.read(13).startswith(b"version https")
    artifact = None if is_lfs_pointer else joblib.load(MODEL_PATH_PICKLE)
else:
    artifact = None

if isinstance(artifact, dict):
    model = artifact["model"]
    model_name = artifact.get("model_name", "bike_rental_model")
    feature_columns = artifact["feature_columns"]
    date_min = pd.Timestamp(artifact["date_min"]).date()
    date_max = pd.Timestamp(artifact["date_max"]).date()
elif artifact is not None:
    raise ValueError("Unsupported model format; expected a model artifact dictionary.")
else:
    model, model_name, feature_columns, date_min, date_max = train_dataset_model(DATASET_PATH)

logging.info("Model loaded: %s", model_name)

# ==========================================
# 2. CREATE FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Bike Rental Demand Prediction API",
    description="Predict bike rental demand using Machine Learning",
    version="1.0"
)

# ==========================================
# 3. DEFINE INPUT DATA
# ==========================================

class RentalInput(BaseModel):

    rental_date: date

    hour: int = Field(ge=0, le=23)

    season: Literal[
        "springer", "summer", "fall", "winter"
    ]

    holiday: Literal["No", "Yes"]

    workingday: Literal[
        "No work", "Working Day"
    ]

    weathersit: Literal[
        "Clear", "Mist", "Light Snow", "Heavy Rain"
    ]

    temp: float = Field(ge=0, le=1)
    atemp: float = Field(ge=0, le=1)
    hum: float = Field(ge=0, le=1)
    windspeed: float = Field(ge=0, le=1)


# ==========================================
# 4. HOME ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {
        "message": "Bike Rental Prediction API is running",
        "model": model_name
    }


# ==========================================
# 5. HEALTH CHECK
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }


# ==========================================
# 6. PREDICTION ENDPOINT
# ==========================================

@app.post("/predict")
def predict(data: RentalInput):

    # Validate supported historical dates
    if not date_min <= data.rental_date <= date_max:
        raise HTTPException(
            status_code=422,
            detail=f"Select a date between {date_min} and {date_max}"
        )

    rental_date = data.rental_date

    # Dataset weekday encoding: Sunday = 0
    input_df = pd.DataFrame([{
        "dteday": pd.Timestamp(rental_date),
        "hr": data.hour,
        "season": data.season,
        "holiday": data.holiday,
        "workingday": data.workingday,
        "weathersit": data.weathersit,
        "temp": data.temp,
        "atemp": data.atemp,
        "hum": data.hum,
        "windspeed": data.windspeed
    }])
    input_df = build_training_features(input_df)[feature_columns]

    # ======================================
    # GENERATE PREDICTION
    # ======================================

    try:

        prediction = model.predict(input_df)[0]

        prediction = max(0, round(float(prediction)))

        return {
            "model": model_name,
            "predicted_rentals": prediction,
            "date": str(rental_date),
            "hour": data.hour,
            "message": "Prediction generated successfully"
        }

    except Exception as error:
        logging.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
