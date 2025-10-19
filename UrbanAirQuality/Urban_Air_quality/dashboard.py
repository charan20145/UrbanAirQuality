# =========================================================
# IMPORTS
# =========================================================
import pandas as pd
import streamlit as st
import plotly.express as px
import requests
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
import numpy as np

# =========================================================
# STREAMLIT CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="🌍 Urban Air Quality Dashboard",
    layout="wide"
)

# =========================================================
# CUSTOM STYLING
# =========================================================
st.markdown("""
    <style>
        body {
            background-color: #F5F7FA;
        }
        .main {
            background: #FFFFFF;
            border-radius: 12px;
            padding: 20px;
        }
        .metric-box {
            background-color: #F0F2F5;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: black;
            font-weight: 600;
            box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
        }
        .metric-box h3 {
            font-size: 18px;
            margin-bottom: 4px;
            color: black;
        }
        .metric-box h2 {
            font-size: 28px;
            color: black;
            font-weight: bold;
        }
        .section-header {
            font-size: 22px;
            font-weight: bold;
            color: #1E3A8A;
            padding-top: 12px;
            padding-bottom: 6px;
            border-bottom: 2px solid #D0E1F9;
        }
        .stDataFrame {
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================
st.markdown(
    "<h1 style='text-align: center; color: #1E3A8A;'>🌍 Urban Air Quality Prediction and Analysis Dashboard</h1>",
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)

mode = st.radio("Select Dashboard Mode:", ["Dataset Analysis", "Real-Time Data Monitoring"], horizontal=True)

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("urban_air_quality_dataset.csv", parse_dates=["DateTime"])
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    return df

# =========================================================
# VISUALIZATION FUNCTION
# =========================================================
def display_charts(df, selected_pollutant):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-header'>Pollution Trend Over Time</div>", unsafe_allow_html=True)
        fig_line = px.line(df, x="DateTime", y=selected_pollutant,
                           title=f"{selected_pollutant} Variation Over Time", markers=True)
        st.plotly_chart(fig_line, use_container_width=True)

        if "TrafficCongestion(%)" in df.columns:
            st.markdown("<div class='section-header'>Traffic Congestion vs Air Quality</div>", unsafe_allow_html=True)
            fig_scatter = px.scatter(df, x="TrafficCongestion(%)", y=selected_pollutant, color="Temperature(C)",
                                     title="Relationship Between Traffic Congestion and AQI",
                                     labels={"TrafficCongestion(%)": "Traffic Congestion (%)"})
            st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        st.markdown("<div class='section-header'>Correlation Heatmap</div>", unsafe_allow_html=True)
        numeric_cols = [c for c in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3",
                                    "Temperature(C)", "Humidity(%)", "WindSpeed(m/s)",
                                    "TrafficCongestion(%)", "AvgVehicleSpeed(km/h)"] if c in df.columns]
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
            st.plotly_chart(fig_corr, use_container_width=True)

        if {"Latitude", "Longitude"}.issubset(df.columns):
            st.markdown("<div class='section-header'>Geospatial Pollution Distribution</div>", unsafe_allow_html=True)
            fig_map = px.density_mapbox(df, lat="Latitude", lon="Longitude", z=selected_pollutant,
                                        radius=10, mapbox_style="open-street-map",
                                        color_continuous_scale="Viridis",
                                        title=f"{selected_pollutant} Concentration by Location")
            st.plotly_chart(fig_map, use_container_width=True)

# =========================================================
# DATASET ANALYSIS MODE
# =========================================================
if mode == "Dataset Analysis":
    df = load_data()
    st.sidebar.header("Filter Options")

    selected_pollutant = st.sidebar.selectbox(
        "Select Pollutant to Analyze:",
        ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
    )

    # Date Filter
    date_range = st.sidebar.date_input(
        "Select Date Range:",
        value=(df["DateTime"].min().date(), df["DateTime"].max().date())
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        df = df[(df["DateTime"].dt.date >= start) & (df["DateTime"].dt.date <= end)]

    # ---------------- OVERVIEW STATISTICS ----------------
    st.markdown("<div class='section-header'>Overview Statistics</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"<div class='metric-box'><h3>Average {selected_pollutant}</h3><h2>{round(df[selected_pollutant].mean(),2)}</h2></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-box'><h3>Maximum {selected_pollutant}</h3><h2>{round(df[selected_pollutant].max(),2)}</h2></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-box'><h3>Minimum {selected_pollutant}</h3><h2>{round(df[selected_pollutant].min(),2)}</h2></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    display_charts(df, selected_pollutant)

    # ---------------- MODEL PERFORMANCE ----------------
    st.markdown("<div class='section-header'>Model Performance and Forecasting</div>", unsafe_allow_html=True)
    features = ["PM10", "NO2", "SO2", "CO", "O3", "Temperature(C)", "Humidity(%)", "WindSpeed(m/s)",
                "TrafficCongestion(%)", "AvgVehicleSpeed(km/h)"]
    X = df[features]
    y = df[selected_pollutant]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
    xgb.fit(X_train, y_train)

    preds_rf = rf.predict(X_test)
    preds_xgb = xgb.predict(X_test)

    rf_rmse = np.sqrt(mean_squared_error(y_test, preds_rf))
    xgb_rmse = np.sqrt(mean_squared_error(y_test, preds_xgb))

    colA, colB = st.columns(2)
    with colA:
        metrics_df = pd.DataFrame({"Model": ["Random Forest", "XGBoost"], "RMSE": [rf_rmse, xgb_rmse]})
        fig_model = px.bar(metrics_df, x="Model", y="RMSE", color="Model", text="RMSE", title="Model RMSE Comparison")
        st.plotly_chart(fig_model, use_container_width=True)
    with colB:
        imp = xgb.feature_importances_
        feature_imp = pd.DataFrame({"Feature": features, "Importance": imp}).sort_values("Importance", ascending=False)
        fig_imp = px.bar(feature_imp, x="Importance", y="Feature", orientation="h", title="Feature Importance (XGBoost)")
        st.plotly_chart(fig_imp, use_container_width=True)

    # ---------------- FORECAST ----------------
    st.markdown("<div class='section-header'>Short-Term AQI Forecast</div>", unsafe_allow_html=True)
    forecast_hours = st.slider("Select forecast period (hours):", 6, 48, 24)
    preds = xgb.predict(X.tail(forecast_hours))
    forecast_df = pd.DataFrame({
        "DateTime": pd.date_range(start=df["DateTime"].iloc[-1], periods=forecast_hours + 1, freq="H")[1:],
        "Predicted": preds
    })
    fig_forecast = px.line(forecast_df, x="DateTime", y="Predicted", markers=True,
                           title=f"Predicted {selected_pollutant} Levels for Next {forecast_hours} Hours")
    st.plotly_chart(fig_forecast, use_container_width=True)

# =========================================================
# REAL-TIME MONITORING MODE
# =========================================================
else:
    st.sidebar.header("Live Air Quality Monitoring")

    API_KEY = "f5eb46470019d6b645570e72a49330a1"
    city = st.sidebar.text_input("Enter City (e.g., London,UK):", "Salford,UK")

    if st.button("Fetch Real-Time Data"):
        try:
            weather_url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
            wdata = requests.get(weather_url, timeout=15).json()
            lat, lon = wdata["coord"]["lat"], wdata["coord"]["lon"]

            aq_url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
            aq_data = requests.get(aq_url, timeout=15).json()
            comp = aq_data["list"][0]["components"]

            st.markdown("<div class='section-header'>Current Air Quality Readings</div>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            col1.markdown(f"<div class='metric-box'><h3>PM2.5</h3><h2>{comp['pm2_5']} µg/m³</h2></div>", unsafe_allow_html=True)
            col2.markdown(f"<div class='metric-box'><h3>PM10</h3><h2>{comp['pm10']} µg/m³</h2></div>", unsafe_allow_html=True)
            col3.markdown(f"<div class='metric-box'><h3>NO₂</h3><h2>{comp['no2']} µg/m³</h2></div>", unsafe_allow_html=True)

            df_real = pd.DataFrame({
                "DateTime": [datetime.now()],
                "PM2.5": [comp["pm2_5"]],
                "PM10": [comp["pm10"]],
                "NO2": [comp["no2"]],
                "SO2": [comp["so2"]],
                "CO": [comp["co"]],
                "O3": [comp["o3"]],
                "Temperature(C)": [wdata["main"]["temp"]],
                "Humidity(%)": [wdata["main"]["humidity"]],
                "WindSpeed(m/s)": [wdata["wind"]["speed"]],
                "Latitude": [lat],
                "Longitude": [lon]
            })

            display_charts(df_real, "PM2.5")

        except Exception as e:
            st.error(f"Error fetching real-time data: {e}")

# =========================================================
# FOOTER
# =========================================================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>Developed for MSDS692 – Data Science Practicum | © 2025 Urban Air Quality Project</p>",
    unsafe_allow_html=True
)
