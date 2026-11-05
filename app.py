import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------
# PAGE CONFIG + THEME
# ---------------------------------------------------------
st.set_page_config(
    page_title="PedalPulse | Bike Rental Analytics",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0b1220; color: #e8eef8; }
[data-testid="stHeader"] { background: rgba(11,18,32,0.85); }
[data-testid="stSidebar"] { background: #111c2e; border-right: 1px solid #26364e; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1500px; }
.hero {
  background: linear-gradient(115deg, #142b45 0%, #173b4c 55%, #174b48 100%);
  border: 1px solid #2c5665; border-radius: 22px; padding: 26px 30px;
  margin-bottom: 20px;
}
.hero-kicker { color: #8de0c4; font-size: 12px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; }
.hero-title { color: #f5f8ff; font-family: 'Manrope', sans-serif; font-size: 34px; font-weight: 800; margin: 5px 0 2px; }
.hero-sub { color: #b9cbdc; font-size: 14px; }
.section-title { font-family: 'Manrope', sans-serif; color: #f1f5fb; font-size: 19px; font-weight: 700; margin: 10px 0 10px; }
.metric-card {
  background: #142238; border: 1px solid #2a3d57; border-radius: 16px;
  padding: 17px 18px; min-height: 112px;
}
.metric-label { color: #9fb2c9; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .8px; }
.metric-value { color: #f4f8ff; font-family: 'Manrope', sans-serif; font-size: 27px; font-weight: 800; margin-top: 7px; }
.metric-note { color: #8de0c4; font-size: 11px; margin-top: 2px; }
div[data-testid="stMetric"] { background: #142238; border: 1px solid #2a3d57; padding: 14px 16px; border-radius: 15px; }
div[data-testid="stMetric"] label { color: #9fb2c9 !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f4f8ff !important; }
.panel {
  background: #111e31; border: 1px solid #263b55; border-radius: 17px;
  padding: 14px 16px 8px; margin-bottom: 14px;
}
div[data-testid="stTabs"] button { color: #c8d5e5; }
.stButton button, .stDownloadButton button {
  background: linear-gradient(90deg, #16a085, #26b99a); color: white;
  border: 0; border-radius: 10px; font-weight: 700;
}
.small-muted { color: #93a7bf; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------
@st.cache_data
def load_data(file_bytes=None):
    if file_bytes is not None:
        from io import BytesIO
        raw = pd.read_csv(BytesIO(file_bytes), na_values=["?"])
    else:
        candidates = [
            BASE_DIR / "Dataset.csv",
            BASE_DIR / "Dataset (1).csv",
            BASE_DIR / "data" / "Dataset.csv"
        ]
        dataset_path = next((p for p in candidates if p.exists()), None)
        if dataset_path is None:
            return None
        raw = pd.read_csv(dataset_path, na_values=["?"])

    raw.columns = raw.columns.str.strip().str.lower()
    raw["dteday"] = pd.to_datetime(
        raw["dteday"], format="%d-%m-%Y", errors="coerce"
    )
    for col in ["cnt", "casual", "registered", "hr", "mnth", "yr",
                "temp", "atemp", "hum", "windspeed", "weekday"]:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=["dteday", "cnt"]).copy()
    raw["year"] = raw["dteday"].dt.year
    raw["month_name"] = raw["dteday"].dt.strftime("%b")
    raw["weekday_name"] = raw["dteday"].dt.day_name()
    raw["date"] = raw["dteday"].dt.date
    raw["hour_label"] = raw["hr"].astype(int).astype(str).str.zfill(2) + ":00"
    return raw.sort_values(["dteday", "hr"]).reset_index(drop=True)

# ---------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------
st.sidebar.markdown("## 🚲 PedalPulse")
st.sidebar.caption("Bike-sharing demand intelligence")
uploaded = st.sidebar.file_uploader("Upload dataset CSV (optional)", type=["csv"])

data = load_data(uploaded.getvalue() if uploaded else None)

if data is None:
    st.markdown("""
    <div class="hero">
      <div class="hero-kicker">Bike-sharing analytics</div>
      <div class="hero-title">PedalPulse Dashboard</div>
      <div class="hero-sub">Upload your bike rental CSV or place it beside app.py as Dataset.csv.</div>
    </div>
    """, unsafe_allow_html=True)
    st.warning("Dataset not found. Upload the project CSV using the sidebar, or add Dataset.csv to the app folder.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("### Filter the data")

min_date = data["dteday"].min().date()
max_date = data["dteday"].max().date()
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

def multiselect_filter(label, column, default_label="All"):
    options = sorted(data[column].dropna().astype(str).unique().tolist())
    return st.sidebar.multiselect(label, options, default=options)

selected_seasons = multiselect_filter("Season", "season")
selected_weather = multiselect_filter("Weather", "weathersit")
selected_working = multiselect_filter("Day type", "workingday")
selected_holidays = multiselect_filter("Holiday", "holiday")
selected_years = st.sidebar.multiselect(
    "Year", sorted(data["year"].dropna().unique().tolist()),
    default=sorted(data["year"].dropna().unique().tolist())
)
hour_range = st.sidebar.slider("Hour of day", 0, 23, (0, 23))

filtered = data[
    (data["date"] >= start_date) &
    (data["date"] <= end_date) &
    (data["season"].astype(str).isin(selected_seasons)) &
    (data["weathersit"].astype(str).isin(selected_weather)) &
    (data["workingday"].astype(str).isin(selected_working)) &
    (data["holiday"].astype(str).isin(selected_holidays)) &
    (data["year"].isin(selected_years)) &
    (data["hr"].between(hour_range[0], hour_range[1]))
].copy()

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown("""
<div class="hero">
  <div class="hero-kicker">Urban mobility · Demand intelligence</div>
  <div class="hero-title">PedalPulse Dashboard</div>
  <div class="hero-sub">Explore rental patterns, understand demand drivers and estimate bike rentals.</div>
</div>
""", unsafe_allow_html=True)

if filtered.empty:
    st.warning("No records match these filters. Adjust the filters in the sidebar.")
    st.stop()

# ---------------------------------------------------------
# KPI CARDS
# ---------------------------------------------------------
total_rentals = int(filtered["cnt"].sum())
avg_hourly = float(filtered["cnt"].mean())
peak_hour = int(filtered.groupby("hr")["cnt"].mean().idxmax())
active_days = filtered["dteday"].nunique()
casual_total = int(filtered["casual"].sum()) if "casual" in filtered else 0
registered_total = int(filtered["registered"].sum()) if "registered" in filtered else 0

st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total rentals", f"{total_rentals:,}")
k2.metric("Average / hour", f"{avg_hourly:,.1f}")
k3.metric("Peak demand hour", f"{peak_hour:02d}:00")
k4.metric("Active days", f"{active_days:,}")
k5.metric("Records analyzed", f"{len(filtered):,}")

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab_overview, tab_patterns, tab_segments, tab_predict, tab_data = st.tabs([
    "📊 Overview", "🕒 Demand patterns", "🌦️ Demand drivers",
    "🤖 Predict rentals", "📁 Data explorer"
])

plot_template = "plotly_dark"
plot_config = {"displayModeBar": False}

with tab_overview:
    left, right = st.columns([1.65, 1])
    with left:
        st.markdown('<div class="section-title">Rental demand over time</div>', unsafe_allow_html=True)
        daily = filtered.groupby("dteday", as_index=False)["cnt"].sum()
        fig = px.line(
            daily, x="dteday", y="cnt",
            labels={"dteday": "Date", "cnt": "Bike rentals"},
            template=plot_template
        )
        fig.update_traces(line=dict(color="#36c9a5", width=2.5), fill="tozeroy",
                          fillcolor="rgba(54,201,165,0.10)")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=15, b=10))
        st.plotly_chart(fig, use_container_width=True, config=plot_config)
    with right:
        st.markdown('<div class="section-title">Rider composition</div>', unsafe_allow_html=True)
        if casual_total + registered_total > 0:
            pie = pd.DataFrame({
                "Rider type": ["Registered", "Casual"],
                "Rentals": [registered_total, casual_total]
            })
            fig = px.pie(
                pie, names="Rider type", values="Rentals", hole=0.68,
                color="Rider type",
                color_discrete_map={"Registered": "#36c9a5", "Casual": "#f5a65b"},
                template=plot_template
            )
            fig.update_layout(height=360, margin=dict(l=5, r=5, t=15, b=10),
                              legend=dict(orientation="h", y=-0.08))
            st.plotly_chart(fig, use_container_width=True, config=plot_config)
        else:
            st.info("Rider composition is unavailable for this selection.")

    st.markdown('<div class="section-title">Average rentals by month</div>', unsafe_allow_html=True)
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly = filtered.groupby("month_name", as_index=False)["cnt"].mean()
    monthly["month_name"] = pd.Categorical(monthly["month_name"], categories=month_order, ordered=True)
    monthly = monthly.sort_values("month_name")
    fig = px.bar(monthly, x="month_name", y="cnt", template=plot_template,
                 labels={"month_name": "Month", "cnt": "Average hourly rentals"})
    fig.update_traces(marker_color="#36c9a5")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True, config=plot_config)

with tab_patterns:
    st.markdown('<div class="section-title">Hourly demand profile</div>', unsafe_allow_html=True)
    hourly = filtered.groupby("hr", as_index=False)["cnt"].mean()
    fig = px.area(hourly, x="hr", y="cnt", template=plot_template,
                  labels={"hr": "Hour of day", "cnt": "Average rentals"})
    fig.update_traces(line_color="#36c9a5", fillcolor="rgba(54,201,165,0.18)")
    fig.update_layout(height=330, xaxis=dict(dtick=1), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True, config=plot_config)

    st.markdown('<div class="section-title">Day of week × hour heatmap</div>', unsafe_allow_html=True)
    weekday_order = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    heat = filtered.pivot_table(
        index="weekday_name", columns="hr", values="cnt",
        aggfunc="mean", observed=False
    ).reindex(weekday_order)
    fig = px.imshow(
        heat, aspect="auto", color_continuous_scale="Tealgrn",
        labels=dict(x="Hour", y="Day of week", color="Avg rentals"),
        template=plot_template
    )
    fig.update_layout(height=390, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True, config=plot_config)

with tab_segments:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Demand by season</div>', unsafe_allow_html=True)
        season_avg = filtered.groupby("season", as_index=False)["cnt"].mean()
        fig = px.bar(season_avg, x="season", y="cnt", template=plot_template,
                     labels={"season": "Season", "cnt": "Average rentals"},
                     color="season",
                     color_discrete_sequence=["#36c9a5", "#4b9ed8", "#f5a65b", "#a78bfa"])
        fig.update_layout(showlegend=False, height=330, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True, config=plot_config)
    with c2:
        st.markdown('<div class="section-title">Demand by weather</div>', unsafe_allow_html=True)
        weather_avg = filtered.groupby("weathersit", as_index=False)["cnt"].mean()
        fig = px.bar(weather_avg, x="weathersit", y="cnt", template=plot_template,
                     labels={"weathersit": "Weather", "cnt": "Average rentals"},
                     color="weathersit", color_discrete_sequence=["#36c9a5", "#4b9ed8", "#f5a65b", "#a78bfa"])
        fig.update_layout(showlegend=False, height=330, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True, config=plot_config)

    st.markdown('<div class="section-title">Temperature and rental demand</div>', unsafe_allow_html=True)
    scatter = filtered.sample(min(2500, len(filtered)), random_state=42)
    fig = px.scatter(
        scatter, x="temp", y="cnt", color="season",
        opacity=0.55, template=plot_template,
        labels={"temp": "Normalized temperature", "cnt": "Bike rentals"}
    )
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True, config=plot_config)

with tab_predict:
    st.markdown('<div class="section-title">Machine learning prediction</div>', unsafe_allow_html=True)
    st.caption("Send a scenario to your FastAPI model. Weather measurements use the dataset's normalized 0–1 scale.")
    try:
        cloud_api_url = st.secrets.get("API_URL")
    except StreamlitSecretNotFoundError:
        cloud_api_url = None
    default_api_url = os.getenv("API_URL") or cloud_api_url or "http://127.0.0.1:8000"
    api_url = st.text_input(
        "FastAPI base URL",
        value=default_api_url,
        help="For local testing use http://127.0.0.1:8000. For deployment, enter your backend URL."
    ).rstrip("/")

    model_path = BASE_DIR / "bike_rental_model.pkl"
    if model_path.exists():
        try:
            with model_path.open("rb") as model_file:
                is_lfs_pointer = model_file.read(13).startswith(b"version https")
            if is_lfs_pointer:
                st.info("The bundled model is a Git LFS pointer. The API will train a fallback model from Dataset.csv.")
                model_min, model_max = date(2011, 1, 1), date(2012, 12, 31)
            else:
                import joblib
                artifact = joblib.load(model_path)
                st.success(f"Model artifact found · {artifact.get('model_name', 'Trained model')}")
                model_min = pd.Timestamp(artifact["date_min"]).date()
                model_max = pd.Timestamp(artifact["date_max"]).date()
        except (OSError, ValueError, KeyError) as error:
            st.warning(f"Could not read the local model artifact: {error}")
            model_min, model_max = date(2011, 1, 1), date(2012, 12, 31)
    else:
        st.info("The dashboard can call your FastAPI service without a local model file.")
        model_min, model_max = date(2011, 1, 1), date(2012, 12, 31)

    with st.form("prediction_form"):
        p1, p2, p3 = st.columns(3)
        with p1:
            pred_date = st.date_input("Rental date", value=date(2012, 6, 15),
                                      min_value=model_min, max_value=model_max)
            pred_hour = st.slider("Hour", 0, 23, 10)
            pred_season = st.selectbox("Season", ["springer", "summer", "fall", "winter"])
        with p2:
            pred_holiday = st.selectbox("Holiday", ["No", "Yes"])
            pred_working = st.selectbox("Working day", ["Working Day", "No work"])
            pred_weather = st.selectbox("Weather", ["Clear", "Mist", "Light Snow", "Heavy Rain"])
        with p3:
            pred_temp = st.slider("Temperature", 0.0, 1.0, 0.5, 0.01)
            pred_atemp = st.slider("Feels-like temperature", 0.0, 1.0, 0.5, 0.01)
            pred_hum = st.slider("Humidity", 0.0, 1.0, 0.5, 0.01)
            pred_wind = st.slider("Wind speed", 0.0, 1.0, 0.2, 0.01)

        predict_clicked = st.form_submit_button("⚡ Generate prediction", use_container_width=True)

    if predict_clicked:
        payload = {
            "rental_date": pred_date.isoformat(),
            "hour": pred_hour,
            "season": pred_season,
            "holiday": pred_holiday,
            "workingday": pred_working,
            "weathersit": pred_weather,
            "temp": pred_temp,
            "atemp": pred_atemp,
            "hum": pred_hum,
            "windspeed": pred_wind
        }
        try:
            with st.spinner("Contacting prediction API..."):
                response = requests.post(f"{api_url}/predict", json=payload, timeout=40)
            if response.ok:
                result = response.json()
                st.success("Prediction completed")
                st.metric("Predicted bike rentals", f"{result['predicted_rentals']:,}")
                st.caption(f"Model: {result.get('model', 'ML model')} · {pred_date} at {pred_hour:02d}:00")
            else:
                st.error(f"API returned {response.status_code}: {response.text}")
        except requests.RequestException as exc:
            st.error(f"Could not reach FastAPI. Check that the backend is running. Details: {exc}")

with tab_data:
    st.markdown('<div class="section-title">Filtered records</div>', unsafe_allow_html=True)
    st.caption(f"Showing {len(filtered):,} records based on the selected filters.")
    display_cols = [
        c for c in ["dteday", "hr", "season", "weathersit", "temp",
                    "hum", "windspeed", "casual", "registered", "cnt"]
        if c in filtered.columns
    ]
    st.dataframe(
        filtered[display_cols].sort_values(["dteday", "hr"], ascending=False),
        use_container_width=True,
        height=420
    )
    csv = filtered[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download filtered data as CSV",
        data=csv,
        file_name="bike_rental_filtered_data.csv",
        mime="text/csv"
    )

st.markdown("---")
st.markdown(
    '<div class="small-muted">PedalPulse · Bike-sharing rental analytics · '
    'Historical dataset dashboard. Predictions are generated by the connected ML API.</div>',
    unsafe_allow_html=True
)
