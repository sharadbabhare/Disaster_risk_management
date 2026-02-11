import joblib
import numpy as np

# Load saved files
model = joblib.load(r"C:\Users\dshek\OneDrive\Desktop\tech\backend\models\heatwave\best_heatwave_model.pkl")
scaler = joblib.load("backend/models/heatwave/scaler.pkl")
feature_columns = joblib.load("backend/models/heatwave/feature_columns.pkl")

# Example input (YOU must give values for all columns in correct order)
# my_input = {
#     "wind_speed": 10.45409633,
#     "cloud_cover": 72.09729216,
#     "pressure_surface_level": 1005.246098,
#     "dew_point": 45.13419688,
#     "uv_index": 0.261182358,
#     "solar_radiation": 220.800292,
#     "max_temperature": 53.81029035,
#     "min_temperature": 45.0229314,
#     "max_humidity": 32.48630154,
#     "min_humidity": 31.49395433,
#     "apparent_temp": 65.67325541,
#     "solar_impact": 7.287695041,
#     "temp_humidity_interaction": 1748.097318,
#     "heat_dome_proxy": 4722.553648,
#     "temp_uv_interaction": 14.05429854,
#     "heat_stress": 17.15269573
# }


# my_input = {
#     "wind_speed": 0.8,
#     "cloud_cover": 5.0,
#     "pressure_surface_level": 1002.0,
#     "dew_point": 31.5,
#     "uv_index": 11.0,
#     "solar_radiation": 950.0,
#     "max_temperature": 46.0,
#     "min_temperature": 34.0,
#     "max_humidity": 78.0,
#     "min_humidity": 45.0,
#     "apparent_temp": 56.0,
#     "solar_impact": 900.0,
#     "temp_humidity_interaction": 3588.0,
#     "heat_dome_proxy": 13000.0,
#     "temp_uv_interaction": 506.0,
#     "heat_stress": 520.0
# }

# my_input = {
#     "wind_speed": 10.45409633,
#     "cloud_cover": 72.09729216,
#     "pressure_surface_level": 1005.246098,
#     "dew_point": 45.13419688,
#     "uv_index": 0.261182358,
#     "solar_radiation": 220.800292,
#     "max_temperature": 53.81029035,
#     "min_temperature": 45.0229314,
#     "max_humidity": 32.48630154,
#     "min_humidity": 31.49395433,
#     "apparent_temp": 65.67325541,
#     "solar_impact": 7.287695041,
#     "temp_humidity_interaction": 1748.097318,
#     "heat_dome_proxy": 4722.553648,
#     "temp_uv_interaction": 14.05429854,
#     "heat_stress": 17.15269573
# }

# my_input = {
#     "wind_speed": 13.5,
#     "cloud_cover": 78.0,
#     "pressure_surface_level": 1008.0,
#     "dew_point": 20.0,
#     "uv_index": 3.2,
#     "solar_radiation": 210.0,
#     "max_temperature": 29.0,
#     "min_temperature": 21.0,
#     "max_humidity": 85.0,
#     "min_humidity": 65.0,
#     "apparent_temp": 31.0,
#     "solar_impact": 150.0,
#     "temp_humidity_interaction": 1500.0,
#     "heat_dome_proxy": 2500.0,
#     "temp_uv_interaction": 95.0,
#     "heat_stress": 160.0
# }


my_input = {
    "wind_speed": 5.2,
    "cloud_cover": 22.0,
    "pressure_surface_level": 994.0,
    "dew_point": 40.5,
    "uv_index": 9.8,
    "solar_radiation": 740.0,
    "max_temperature": 47.0,
    "min_temperature": 39.0,
    "max_humidity": 42.0,
    "min_humidity": 21.0,
    "apparent_temp": 57.0,
    "solar_impact": 640.0,
    "temp_humidity_interaction": 2100.0,
    "heat_dome_proxy": 6900.0,
    "temp_uv_interaction": 460.0,
    "heat_stress": 420.0
}











# Arrange in exact training order
import pandas as pd

X_df = pd.DataFrame([my_input], columns=feature_columns)
X_scaled = scaler.transform(X_df)


# Scale
# X_scaled = scaler.transform(X)

# Predict
pred = model.predict(X_scaled)[0]
prob = model.predict_proba(X_scaled)[0][1]
# threshold = 0.30   # you can change this (0.3 / 0.4 / 0.5)

# custom_pred = 1 if prob >= threshold else 0

# print("Normal Prediction (0.5):", int(pred))
# print("Probability_heatwave:", float(prob))
# print(f"Custom Prediction ({threshold}):", int(custom_pred))

print("Prediction:", int(pred))
print("Probability_heatwave:", float(prob))
