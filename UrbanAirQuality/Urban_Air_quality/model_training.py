# =========================================================
# 1. IMPORTS
# =========================================================
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib

# =========================================================
# 2. LOAD AND PREPROCESS DATA
# =========================================================
print("📂 Loading dataset...")
df = pd.read_csv("urban_air_quality_dataset.csv", parse_dates=["DateTime"])
df["DateTime"] = pd.to_datetime(df["DateTime"])
df = df.sort_values(by="DateTime")

# Handle missing values
df = df.dropna()

# Define features and target
features = ["PM10", "NO2", "SO2", "CO", "O3", "Temperature(C)", "Humidity(%)",
            "WindSpeed(m/s)", "TrafficCongestion(%)", "AvgVehicleSpeed(km/h)"]
target = "PM2.5"

X = df[features]
y = df[target]

# Normalize for deep learning
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
print("✅ Data preprocessing completed.\n")

# =========================================================
# 3. RANDOM FOREST MODEL
# =========================================================
print("🌳 Training Random Forest model...")
rf_model = RandomForestRegressor(n_estimators=150, random_state=42)
rf_model.fit(X_train, y_train.ravel())
rf_preds = rf_model.predict(X_test)

# Evaluate Random Forest
rf_rmse = sqrt(mean_squared_error(y_test, rf_preds))
rf_mae = mean_absolute_error(y_test, rf_preds)
rf_r2 = r2_score(y_test, rf_preds)

# =========================================================
# 4. XGBOOST MODEL
# =========================================================
print("⚙️ Training XGBoost model...")
xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train.ravel())
xgb_preds = xgb_model.predict(X_test)

# Evaluate XGBoost
xgb_rmse = sqrt(mean_squared_error(y_test, xgb_preds))
xgb_mae = mean_absolute_error(y_test, xgb_preds)
xgb_r2 = r2_score(y_test, xgb_preds)

# =========================================================
# 5. LSTM MODEL (Deep Learning)
# =========================================================
print("🧠 Training LSTM model (this may take a few minutes)...")
X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_scaled, test_size=0.2, random_state=42)

lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dense(16, activation='relu'),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse')
history = lstm_model.fit(X_train_lstm, y_train_lstm, validation_split=0.2, epochs=20, batch_size=32, verbose=1)

lstm_preds = lstm_model.predict(X_test_lstm)
lstm_rmse = sqrt(mean_squared_error(y_test_lstm, lstm_preds))
lstm_mae = mean_absolute_error(y_test_lstm, lstm_preds)
lstm_r2 = r2_score(y_test_lstm, lstm_preds)

# =========================================================
# 6. COMPARE MODEL PERFORMANCE (Plotly)
# =========================================================
print("📊 Comparing model performances...")
metrics_df = pd.DataFrame({
    "Model": ["Random Forest", "XGBoost", "LSTM"],
    "RMSE": [rf_rmse, xgb_rmse, lstm_rmse],
    "MAE": [rf_mae, xgb_mae, lstm_mae],
    "R²": [rf_r2, xgb_r2, lstm_r2]
})

print("\n=== MODEL PERFORMANCE SUMMARY ===")
print(metrics_df)

# Bar Chart: RMSE Comparison
fig_rmse = px.bar(metrics_df, x="Model", y="RMSE", color="Model",
                  title="Model Performance Comparison (RMSE)",
                  text_auto=True)
fig_rmse.show()

# Line Chart: Predicted vs Actual for XGBoost
fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(y=y_test.ravel(), mode="lines", name="Actual"))
fig_pred.add_trace(go.Scatter(y=xgb_preds, mode="lines", name="Predicted (XGBoost)"))
fig_pred.update_layout(title="Actual vs Predicted AQI (XGBoost)", xaxis_title="Samples", yaxis_title="PM2.5")
fig_pred.show()

# Feature Importance (XGBoost)
importance = xgb_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
fig_imp = px.bar(feature_importance_df.sort_values("Importance", ascending=False),
                 x="Importance", y="Feature", orientation="h",
                 title="Feature Importance (XGBoost)")
fig_imp.show()

# =========================================================
# 7. SAVE TRAINED MODELS
# =========================================================
joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(xgb_model, "xgboost_model.pkl")
lstm_model.save("lstm_model.h5")
joblib.dump(scaler, "scaler.pkl")

print("\n Models trained and saved successfully!")
print(metrics_df)
