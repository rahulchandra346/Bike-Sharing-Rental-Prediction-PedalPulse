# PedalPulse Bike Rental Prediction

PedalPulse is a Streamlit dashboard for exploring historical bike-sharing demand, with a FastAPI service that predicts hourly rentals.

## Requirements

- Python 3.10 or later
- The project dependencies in `requirements.txt`

## Run locally

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Start the prediction API in one terminal:

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Start the dashboard in another terminal:

```powershell
python -m streamlit run app.py
```

Open the dashboard at <http://localhost:8501>. The prediction tab connects to the API at <http://127.0.0.1:8000> by default.

## Model

When no usable model artifact is present, the API trains an Extra Trees fallback from `Dataset.csv` at startup. To use a trained model artifact, place `bike_rental_model.joblib` inside `api/`.

## Deployment

The Streamlit dashboard can be deployed from the repository root with `app.py` as the entry point. The FastAPI service is isolated in `api/` with its own minimal dependencies and dataset so it can be deployed separately on Vercel with `api/` as the project root. Set the dashboard's `API_URL` environment variable to the Vercel deployment URL.
