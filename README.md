# 🌍 Urban Air Quality Prediction and Analysis Dashboard

## 📘 Project Overview
The **Urban Air Quality Dashboard** is a data-driven project designed to **analyze, visualize, and forecast air quality (AQI)** across cities using **multi-source environmental, meteorological, and traffic data**.  

By leveraging **machine learning and deep learning models**, this dashboard predicts pollution levels, supports **real-time monitoring**, and provides valuable insights for decision-making in **urban sustainability, climate resilience, and public health management**.

Developed as part of the **MSDS692 – Data Science Practicum**, this project showcases the integration of data analytics, visualization, and AI-driven forecasting for smarter urban environments.

---

## 🎯 Objectives
- To integrate multi-source environmental and meteorological data for urban AQI prediction.  
- To design an **interactive Streamlit dashboard** for visualizing air quality patterns.  
- To compare predictive performance across **Random Forest, XGBoost, and LSTM** models.  
- To enable **real-time air quality monitoring** via the OpenWeatherMap API.  
- To promote awareness and provide data-driven decision support for pollution control.

---

## ✨ Key Features
✅ **Dataset Analysis Mode**
- Analyze historical air quality data with interactive filters and date ranges.  
- Visualize pollutant patterns, correlations, and spatial intensity.  
- Train and compare predictive models for AQI forecasting.  

✅ **Real-Time Monitoring Mode**
- Fetch **live AQI and weather data** from the OpenWeatherMap API.  
- Display current pollutant concentrations and weather parameters.  
- Generate **short-term forecast charts** for the next 12–48 hours.  

✅ **Visualization Highlights**
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

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<YOUR-USERNAME>/UrbanAirQuality.git
cd UrbanAirQuality
2️⃣ Create and Activate a Virtual Environment
python -m venv .venv
.venv\Scripts\activate
3️⃣ Install Required Libraries
pip install -r requirements.txt
Model Training
python model_training.pyrandom_forest_model.pkl
xgboost_model.pkl
lstm_model.h5
scaler.pkl


 Running the Dashboard
streamlit run dashboard.py

Project Folder Structure
UrbanAirQuality/
│
├── dashboard.py                 # Streamlit dashboard app
├── model_training.py            # Model training and evaluation script
├── urban_air_quality_dataset.csv# Dataset
├── requirements.txt             # Python dependencies
├── README.md                    # Documentation
├── models/                      # Saved trained models
└── visuals/                     # Screenshots and figures (optional)


this my github readme full script,update it to work correctry and proffesionally
