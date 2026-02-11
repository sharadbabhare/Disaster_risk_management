import joblib
import pandas as pd

# Load landslide files
model = joblib.load("backend/models/landslide/landslide_model.pkl")
scaler = joblib.load("backend/models/landslide/landslide_scaler.pkl")
feature_columns = joblib.load("backend/models/landslide/landslide_feature_columns.pkl")

# ✅ Dummy landslide input (change values)
my_input = {
    "Elevation": 250,        # meters
    "Slope": 28,             # degrees
    "Curvature": 3.5,        # 1/m
    "Precipitation": 100,     # mm
    "Earthquake": 4.5,       # richter
    "temperature": 22,       # °C
    "moisture": 65,          # %
    "NDVI": 0.35,            # -1 to 1
    "NDWI": 0.28,            # -1 to 1
    "Lithology": 2           # dummy category
}

# Arrange columns
X_df = pd.DataFrame([my_input], columns=feature_columns)
X_scaled = scaler.transform(X_df)

pred = int(model.predict(X_scaled)[0])
prob = float(model.predict_proba(X_scaled)[0][1])

print("Prediction:", pred)
print("Probability_landslide:", prob)
