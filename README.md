# PedalPulse — Bike-Sharing Rental Prediction

PedalPulse is a bike-sharing demand analytics and prediction project. It helps users explore historical hourly rental patterns and estimate how many bikes may be rented for a selected date, hour, and set of conditions.

## Project objective

Bike-sharing operators need to keep enough bicycles available where and when riders need them. Underestimating demand can lead to unavailable bikes and longer waits; overestimating it can leave bikes idle. PedalPulse uses historical demand and contextual factors such as time, season, weather, and working-day status to support demand planning.

The project brief sets out a broader machine-learning workflow: explore and clean the data, engineer useful features, compare regression models, tune and evaluate them, and deploy a selected model. The current repository provides the interactive analytics dashboard and a working prediction API. The API currently uses a dataset-based nearest-neighbor baseline; the model-comparison and hyperparameter-tuning work in the brief remains future work (see [Model status and next steps](#model-status-and-next-steps)).

## Live application

- **Dashboard:** [Open PedalPulse](https://bike-sharing-rental-prediction-pedalpulse-fhfnrsj75xtdbdcvdzik.streamlit.app/)
- **API base URL:** [Open the Vercel project URL](https://bike-sharing-rental-peda-psi.vercel.app/) — redirects to the dashboard
- **API health:** [`/health`](https://bike-sharing-rental-peda-psi.vercel.app/health)
- **Interactive API documentation:** [`/docs`](https://bike-sharing-rental-peda-psi.vercel.app/docs)

The API also exposes `POST /predict`.

## What the dashboard does

The Streamlit dashboard reads the included hourly dataset and provides:

- Summary metrics for rentals, average hourly demand, the peak-demand hour, active days, and records.
- Interactive filters for date range, season, weather, day type, holiday, year, and hour.
- Visualizations of demand over time and by hour, month, weekday, season, weather, and temperature.
- A data explorer for filtered records.
- A prediction form that sends a selected scenario to the FastAPI service.
- Optional CSV upload to explore another compatible dataset.

## Data

`Dataset.csv` contains hourly bike-sharing observations from 2011–2012. The dashboard and API expect the project dataset's column names and category values. Main fields include:

| Field | Description |
|---|---|
| `dteday`, `hr` | Date and hour of the observation |
| `season`, `yr`, `mnth`, `weekday` | Calendar and seasonal context |
| `holiday`, `workingday` | Holiday and working-day indicators |
| `weathersit` | Weather category |
| `temp`, `atemp`, `hum`, `windspeed` | Normalized weather measurements |
| `casual`, `registered` | Rental counts by user type |
| `cnt` | Total hourly rentals; prediction target |

Weather measurements are normalized on a 0–1 scale. Uploaded CSVs should follow the same schema and category conventions for the dashboard and prediction workflow to work as intended.

## Prediction approach

The FastAPI service loads the bundled dataset when it starts. For each request, its nearest-neighbor baseline ranks historical hourly records by similarity to the requested date, hour, season, weather, holiday and working-day status, and normalized weather measurements. It combines the rental counts from the closest records into a demand estimate.

This is a functional baseline, not a claim that the project brief's proposed Decision Tree, Random Forest, and Gradient Boosting comparison or hyperparameter search has been completed. The current service does not return a reported MAE, MSE, RMSE, or R² score.

## Run locally

Use Python 3.12 (the version configured for the hosted apps). From the repository root, create and activate a virtual environment, then install the dashboard dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

In one terminal, start the API:

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

In a second terminal, from the repository root, start the dashboard:

```powershell
python -m streamlit run app.py
```

Open the dashboard at <http://localhost:8501>. By default, its prediction form calls the API at <http://127.0.0.1:8000>. Set `API_URL` to use a different API:

```powershell
$env:API_URL = "https://bike-sharing-rental-peda-psi.vercel.app"
python -m streamlit run app.py
```

## API

The API accepts a JSON request at `POST /predict`:

```json
{
  "rental_date": "2012-06-15",
  "hour": 10,
  "season": "summer",
  "holiday": "No",
  "workingday": "Working Day",
  "weathersit": "Clear",
  "temp": 0.5,
  "atemp": 0.5,
  "hum": 0.5,
  "windspeed": 0.2
}
```

The date must be within the training dataset's date range. The response includes the predicted rental count, date, hour, and model name. The API also provides `GET /health` and `GET /docs`.

Install API dependencies separately when running only the API:

```powershell
python -m pip install -r api\requirements.txt
```

## Deployment

The live app uses two services:

1. **Streamlit Community Cloud** hosts `app.py` from the repository root.
2. **Vercel** hosts the FastAPI service with `api/` as its project root. `api/Dataset.csv` is included so the API can load its training data within that deployment.

The dashboard's Streamlit Cloud secret `API_URL` points to the Vercel API. To configure a separate deployment, set `API_URL` to its API base URL in that app's secrets. Locally, an `API_URL` environment variable takes precedence over Streamlit secrets.

## Repository layout

```text
.
├── app.py                 # Streamlit dashboard
├── main.py                # Local API entry point
├── Dataset.csv            # Dashboard and prediction data
├── requirements.txt       # Dashboard/local project dependencies
└── api/
    ├── main.py             # FastAPI app and nearest-neighbor baseline
    ├── Dataset.csv         # Dataset packaged for Vercel
    └── requirements.txt    # Minimal API deployment dependencies
```

## Model status and next steps

The project brief identifies these next steps for a more complete modeling study:

1. Audit data quality, including missing values, duplicates, and outliers; document any cleaning decisions.
2. Build and compare Decision Tree, Random Forest, and Gradient Boosting regressors using an appropriate train/test split and cross-validation.
3. Tune promising models, then report and compare MAE, MSE, RMSE, and R² on held-out data.
4. Deploy the selected, evaluated model and monitor its prediction quality over time.

Until that evaluation is completed, use predictions as a planning aid rather than a guarantee of actual rental demand.

## Contributors

- [rahulchandra346](https://github.com/rahulchandra346) — Project owner
