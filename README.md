# 🌍 Urban Air Quality Prediction and Analysis Dashboard

##  Project Overview
The **Urban Air Quality Dashboard** is a data-driven project designed to **analyze, visualize, and forecast air quality (AQI)** across cities using **multi-source environmental, meteorological, and traffic data**.  

By leveraging **machine learning and deep learning models**, this dashboard predicts pollution levels, supports **real-time monitoring**, and provides valuable insights for decision-making in **urban sustainability, climate resilience, and public health management**.

Developed as part of the **MSDS692 – Data Science Practicum**, this project showcases the integration of data analytics, visualization, and AI-driven forecasting for smarter urban environments.

---

##  Objectives
- To integrate multi-source environmental and meteorological data for urban AQI prediction.  
- To design an **interactive Streamlit dashboard** for visualizing air quality patterns.  
- To compare predictive performance across **Random Forest, XGBoost, and LSTM** models.  
- To enable **real-time air quality monitoring** via the OpenWeatherMap API.  
- To promote awareness and provide data-driven decision support for pollution control.

---

##  Key Features
 **Dataset Analysis Mode**
- Analyze historical air quality data with interactive filters and date ranges.  
- Visualize pollutant patterns, correlations, and spatial intensity.  
- Train and compare predictive models for AQI forecasting.  

 **Real-Time Monitoring Mode**
- Fetch **live AQI and weather data** from the OpenWeatherMap API.  
- Display current pollutant concentrations and weather parameters.  
- Generate **short-term forecast charts** for the next 12–48 hours.  

 **Visualization Highlights**
- AQI trends over time  
- Correlation heatmaps  
- Traffic congestion vs pollution scatterplots  
- Pollution hotspot (geospatial) maps  
- Feature importance visualization  
- Model comparison and performance bar charts  
- Forecast line chart showing predicted AQI trends  

---

##  Tech Stack

**Languages:** Python 3.9+  
**Libraries & Frameworks:**  
- `pandas`, `numpy`, `scikit-learn`, `xgboost`, `tensorflow`, `plotly`, `streamlit`, `requests`, `joblib`  
**Tools:**  
- Visual Studio Code / Jupyter Notebook  
- Streamlit for dashboard deployment  
- OpenWeatherMap API for live data  
- Git & GitHub for version control

---
    
## Dataset Description

The project uses a combination of historical air quality, meteorological, and traffic datasets to predict the Air Quality Index (AQI).

 Data Sources:

Air Quality Data: Contains daily concentrations of major pollutants (PM2.5, PM10, NO₂, CO, SO₂, O₃).

Meteorological Data: Includes temperature, humidity, wind speed, and pressure readings.

Traffic and Urban Data: Captures vehicle density or congestion levels from selected regions.

---

## Dataset Features:
Feature	Description
Date	Timestamp of data collection
City	Name of the monitoring city
PM2.5	Fine particulate matter concentration
PM10	Coarse particulate matter concentration
NO₂	Nitrogen dioxide concentration
CO	Carbon monoxide concentration
O₃	Ozone level
Temperature	Ambient temperature (°C)
Humidity	Relative humidity (%)
Wind Speed	Wind velocity (m/s)
AQI	Computed Air Quality Index label

Data was preprocessed to remove missing values, scale features, and merge multiple data sources into one standardized dataset used for model training and prediction.

 Dashboard Explanation

The Streamlit Dashboard is designed for interactive exploration and forecasting of air quality trends.

Main Sections:

Home Page: Introduction and project overview.


Dataset Analysis Mode:

Users can upload or explore the historical dataset.

Displays pollutant distributions, time trends, and correlations.

Model Comparison Section:

Visualizes performance metrics of Random Forest, XGBoost, and LSTM models.

Displays RMSE, MAE, and R² values side-by-side.

Real-Time Monitoring Mode:

Fetches current AQI and weather conditions from OpenWeatherMap API.

Predicts next 12–48 hours of AQI values.

Forecast Visualization:

Interactive line chart showing predicted vs. actual AQI over time.

The dashboard provides a clean, user-friendly interface with dynamic charts powered by Plotly and Matplotlib.

Results and Findings
 Model Performance Summary:
Model	RMSE	MAE	R²
Random Forest	6.85	5.20	0.89
XGBoost	6.12	4.75	0.91
LSTM (Deep Learning)	5.84	4.32	0.93

The LSTM model achieved the highest accuracy, demonstrating strong capability in capturing temporal dependencies in AQI time-series data.

 Key Insights:

PM2.5 and PM10 are the most influential pollutants affecting AQI.

AQI tends to worsen during low wind speed and high humidity conditions.

Model integration with live data enables real-time forecasting and policy planning support.

What Each Code File Does
model_training.py

Loads and preprocesses the dataset.

Splits data into training and testing sets.

Trains and saves three models:

Random Forest (random_forest_model.pkl)

XGBoost (xgboost_model.pkl)

LSTM (lstm_model.h5)

Evaluates each model using RMSE, MAE, and R².

Saves a scaler.pkl for consistent normalization during prediction.

dashboard.py
Launches the Streamlit dashboard interface.

Allows dataset upload or exploration from local files.

Displays pollutant trends, EDA charts, and correlation heatmaps.

Loads pre-trained models for on-demand prediction.

Integrates OpenWeatherMap API to fetch live data.

Visualizes real-time AQI forecasts (next 12–48 hours).

requirements.txt
Lists all dependencies required to run the project (e.g., pandas, streamlit, xgboost, etc.).

urban_air_quality_dataset.csv
The cleaned and structured dataset used for both training and visualization.

models
Directory containing all saved trained models and the scaler file.

visuals/
Contains plots, screenshots, or visual artifacts used in the dashboard and report.

