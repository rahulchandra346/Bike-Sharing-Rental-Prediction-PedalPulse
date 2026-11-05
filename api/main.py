import csv
import heapq
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "Dataset.csv"
FEATURE_COLUMNS = [
    "season", "yr", "holiday", "workingday", "weathersit", "temp", "atemp",
    "hum", "windspeed", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "month_sin", "month_cos", "day_of_month", "day_of_year"
]


@dataclass(frozen=True)
class TrainingRow:
    rental_date: date
    hour: int
    season: str
    holiday: str
    workingday: str
    weathersit: str
    temp: float
    atemp: float
    hum: float
    windspeed: float
    rentals: float


class RentalInput(BaseModel):
    rental_date: date
    hour: int = Field(ge=0, le=23)
    season: Literal["springer", "summer", "fall", "winter"]
    holiday: Literal["No", "Yes"]
    workingday: Literal["No work", "Working Day"]
    weathersit: Literal["Clear", "Mist", "Light Snow", "Heavy Rain"]
    temp: float = Field(ge=0, le=1)
    atemp: float = Field(ge=0, le=1)
    hum: float = Field(ge=0, le=1)
    windspeed: float = Field(ge=0, le=1)


def load_training_rows(dataset_path: Path) -> list[TrainingRow]:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Training dataset not found: {dataset_path}")

    rows = []
    with dataset_path.open(encoding="utf-8", newline="") as dataset_file:
        for row in csv.DictReader(dataset_file):
            try:
                rows.append(TrainingRow(
                    rental_date=datetime.strptime(row["dteday"], "%d-%m-%Y").date(),
                    hour=int(row["hr"]),
                    season=row["season"],
                    holiday=row["holiday"],
                    workingday=row["workingday"],
                    weathersit=row["weathersit"],
                    temp=float(row["temp"]),
                    atemp=float(row["atemp"]),
                    hum=float(row["hum"]),
                    windspeed=float(row["windspeed"]),
                    rentals=float(row["cnt"])
                ))
            except (KeyError, TypeError, ValueError):
                continue

    if not rows:
        raise ValueError(f"No valid training records found in {dataset_path}")
    return rows


class RentalDemandModel:
    def __init__(self, rows: list[TrainingRow]):
        self.rows = rows

    @staticmethod
    def _cyclic_distance(left: int, right: int, cycle: int) -> int:
        difference = abs(left - right)
        return min(difference, cycle - difference)

    def predict(self, scenario: RentalInput) -> float:
        target_weekday = (scenario.rental_date.weekday() + 1) % 7

        def distance(row: TrainingRow) -> float:
            row_weekday = (row.rental_date.weekday() + 1) % 7
            row_month = row.rental_date.month
            month_distance = self._cyclic_distance(
                row_month, scenario.rental_date.month, 12
            )
            hour_distance = self._cyclic_distance(row.hour, scenario.hour, 24)
            weekday_distance = self._cyclic_distance(row_weekday, target_weekday, 7)
            score = (hour_distance / 3) ** 2
            score += (weekday_distance / 3.5) ** 2 * 0.3
            score += (month_distance / 6) ** 2 * 0.5
            score += 0.25 if row.rental_date.year != scenario.rental_date.year else 0
            score += 1.5 if row.season != scenario.season else 0
            score += 1.0 if row.weathersit != scenario.weathersit else 0
            score += 0.75 if row.workingday != scenario.workingday else 0
            score += 0.5 if row.holiday != scenario.holiday else 0
            score += ((row.temp - scenario.temp) / 0.2) ** 2 * 0.05
            score += ((row.atemp - scenario.atemp) / 0.2) ** 2 * 0.025
            score += ((row.hum - scenario.hum) / 0.25) ** 2 * 0.025
            score += ((row.windspeed - scenario.windspeed) / 0.2) ** 2 * 0.025
            return score

        nearest = heapq.nsmallest(
            25,
            ((distance(row), row) for row in self.rows),
            key=lambda item: item[0]
        )
        weighted_rentals = sum(
            row.rentals / (score + 0.05) for score, row in nearest
        )
        total_weight = sum(1 / (score + 0.05) for score, _ in nearest)
        return weighted_rentals / total_weight


training_rows = load_training_rows(DATASET_PATH)
model = RentalDemandModel(training_rows)
model_name = "Dataset-trained nearest-neighbor model"
date_min = min(row.rental_date for row in training_rows)
date_max = max(row.rental_date for row in training_rows)
feature_columns = FEATURE_COLUMNS

app = FastAPI(
    title="Bike Rental Demand Prediction API",
    description="Predict bike rental demand using historical ride data",
    version="1.0"
)


@app.get("/")
def home():
    return RedirectResponse(
        "https://bike-sharing-rental-prediction-pedalpulse-fhfnrsj75xtdbdcvdzik.streamlit.app/"
    )


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict(data: RentalInput):
    if not date_min <= data.rental_date <= date_max:
        raise HTTPException(
            status_code=422,
            detail=f"Select a date between {date_min} and {date_max}"
        )

    prediction = max(0, round(model.predict(data)))
    return {
        "model": model_name,
        "predicted_rentals": prediction,
        "date": str(data.rental_date),
        "hour": data.hour,
        "message": "Prediction generated successfully"
    }
