# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# import joblib
# import pandas as pd
# import requests

# # Load model
# model = joblib.load("models/heatwave/best_heatwave_model.pkl")
# scaler = joblib.load("models/heatwave/scaler.pkl")
# feature_columns = joblib.load("models/heatwave/feature_columns.pkl")


# # ✅ Load landslide model
# landslide_model = joblib.load("models/landslide/landslide_model.pkl")
# landslide_scaler = joblib.load("models/landslide/landslide_scaler.pkl")
# landslide_features = joblib.load("models/landslide/landslide_feature_columns.pkl")


# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # -------------------------
# # Fetch weather
# # -------------------------
# def fetch_weather(lat, lon):
#     weatherURL = (
#         f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
#         "&daily=temperature_2m_min,temperature_2m_max,relative_humidity_2m_min,relative_humidity_2m_max,uv_index_max,shortwave_radiation_sum"
#         "&hourly=dew_point_2m,pressure_msl"
#         "&current=cloud_cover,wind_speed_10m"
#     )
#     return requests.get(weatherURL, timeout=20).json()



#                      #INSERT ELEVATION

# # def fetch_elevation(lat, lon):
# #     url = f"https://api.opentopodata.org/v1/srtm90m?locations={lat},{lon}"
# #     data = requests.get(url, timeout=20).json()
# #     return float(data["results"][0]["elevation"])              
# # -------------------------
# # Build features
# # -------------------------
# def build_features(w):
#     min_temp = float(w["daily"]["temperature_2m_min"][0])
#     max_temp = float(w["daily"]["temperature_2m_max"][0])
#     min_hum = float(w["daily"]["relative_humidity_2m_min"][0])
#     max_hum = float(w["daily"]["relative_humidity_2m_max"][0])
#     uv_index = float(w["daily"]["uv_index_max"][0])
#     solar_radiation = float(w["daily"]["shortwave_radiation_sum"][0])

#     cloud_cover = float(w["current"]["cloud_cover"])
#     wind_speed_kmh = float(w["current"]["wind_speed_10m"])

#     pressure = float(w["hourly"]["pressure_msl"][0])
#     dew_point = float(w["hourly"]["dew_point_2m"][0])

#     # km/h -> m/s
#     wind_speed = wind_speed_kmh / 3.6

#     # Derived features (same logic as your training)
#     apparent_temp = max_temp + (0.1 * dew_point)
#     solar_impact = solar_radiation * (1 - cloud_cover / 100)

#     temp_humidity_interaction = max_temp * max_hum
#     heat_dome_proxy = max_temp * (1015 - pressure)
#     temp_uv_interaction = max_temp * uv_index
#     heat_stress = apparent_temp * uv_index

#     return {
#         "wind_speed": wind_speed,
#         "cloud_cover": cloud_cover,
#         "pressure_surface_level": pressure,
#         "dew_point": dew_point,
#         "uv_index": uv_index,
#         "solar_radiation": solar_radiation,
#         "max_temperature": max_temp,
#         "min_temperature": min_temp,
#         "max_humidity": max_hum,
#         "min_humidity": min_hum,
#         "apparent_temp": apparent_temp,
#         "solar_impact": solar_impact,
#         "temp_humidity_interaction": temp_humidity_interaction,
#         "heat_dome_proxy": heat_dome_proxy,
#         "temp_uv_interaction": temp_uv_interaction,
#         "heat_stress": heat_stress
#     }
# def predict_landslide(features_dict: dict):
#     """
#     features_dict must contain keys matching landslide_features
#     """
#     X_df = pd.DataFrame([features_dict], columns=landslide_features)
#     X_scaled = landslide_scaler.transform(X_df)

#     prob = float(landslide_model.predict_proba(X_scaled)[0][1])
#     pred_default = int(prob >= 0.5)

#     # custom threshold
#     threshold = 0.30
#     pred_custom = int(prob >= threshold)

#     return {
#         "probability_landslide": round(prob, 4),
#         "probability_percent": round(prob * 100, 2),
#         "prediction_default_0_5": pred_default,
#         "prediction_custom_0_30": pred_custom
#     }


# @app.get("/")
# def home():
#     return {"status": "Heatwave Predictor API running ✅"}

# @app.get("/predict-live")
# def predict_live(lat: float, lon: float):
#     w = fetch_weather(lat, lon)

#     features = build_features(w)

#     X_df = pd.DataFrame([features], columns=feature_columns)
#     X_scaled = scaler.transform(X_df)

    # prob = float(model.predict_proba(X_scaled)[0][1])

    # pred_default = int(prob >= 0.5)
    # threshold = 0.30
    # pred_custom = int(prob >= threshold)

    # return {
    #     "lat": lat,
    #     "lon": lon,
    #     "probability__heatwave": round(prob, 4),
    #     "probability_percent": round(prob * 100, 2),
    #     "prediction_default_0_5": pred_default,
    #     "prediction_custom_0_30": pred_custom,
    #     "features_used": features
    # }





# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# import joblib
# import pandas as pd
# import requests

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # -------------------------
# # LOAD MODELS
# # -------------------------
# # Heatwave
# heatwave_model = joblib.load("models/heatwave/best_heatwave_model.pkl")
# heatwave_scaler = joblib.load("models/heatwave/scaler.pkl")
# heatwave_features = joblib.load("models/heatwave/feature_columns.pkl")

# # Landslide
# landslide_model = joblib.load("models/landslide/landslide_model.pkl")
# landslide_scaler = joblib.load("models/landslide/landslide_scaler.pkl")
# landslide_features = joblib.load("models/landslide/landslide_feature_columns.pkl")

# # -------------------------
# # APIs
# # -------------------------
# def fetch_weather(lat, lon):
#     url = (
#         f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
#         "&daily=temperature_2m_min,temperature_2m_max,relative_humidity_2m_min,relative_humidity_2m_max,uv_index_max,shortwave_radiation_sum"
#         "&hourly=dew_point_2m,pressure_msl,rain"
#         "&current=cloud_cover,wind_speed_10m"
#     )
#     return requests.get(url, timeout=20).json()

# def fetch_elevation(lat, lon):
#     url = f"https://api.opentopodata.org/v1/srtm90m?locations={lat},{lon}"
#     data = requests.get(url, timeout=20).json()
#     return float(data["results"][0]["elevation"])

# # -------------------------
# # FEATURE BUILDERS
# # -------------------------
# def build_heatwave_features(w):
#     min_temp = float(w["daily"]["temperature_2m_min"][0])
#     max_temp = float(w["daily"]["temperature_2m_max"][0])
#     min_hum = float(w["daily"]["relative_humidity_2m_min"][0])
#     max_hum = float(w["daily"]["relative_humidity_2m_max"][0])

#     uv_index = float(w["daily"]["uv_index_max"][0])
#     solar_radiation = float(w["daily"]["shortwave_radiation_sum"][0])

#     cloud_cover = float(w["current"]["cloud_cover"])
#     wind_speed_kmh = float(w["current"]["wind_speed_10m"])

#     pressure = float(w["hourly"]["pressure_msl"][0])
#     dew_point = float(w["hourly"]["dew_point_2m"][0])

#     # km/h -> m/s
#     wind_speed = wind_speed_kmh / 3.6

#     # derived
#     apparent_temp = max_temp + (0.1 * dew_point)
#     solar_impact = solar_radiation * (1 - cloud_cover / 100)

#     temp_humidity_interaction = max_temp * max_hum
#     heat_dome_proxy = max_temp * (1015 - pressure)
#     temp_uv_interaction = max_temp * uv_index
#     heat_stress = apparent_temp * uv_index

#     return {
#         "wind_speed": wind_speed,
#         "cloud_cover": cloud_cover,
#         "pressure_surface_level": pressure,
#         "dew_point": dew_point,
#         "uv_index": uv_index,
#         "solar_radiation": solar_radiation,
#         "max_temperature": max_temp,
#         "min_temperature": min_temp,
#         "max_humidity": max_hum,
#         "min_humidity": min_hum,
#         "apparent_temp": apparent_temp,
#         "solar_impact": solar_impact,
#         "temp_humidity_interaction": temp_humidity_interaction,
#         "heat_dome_proxy": heat_dome_proxy,
#         "temp_uv_interaction": temp_uv_interaction,
#         "heat_stress": heat_stress
#     }

# def predict_heatwave(features):
#     X_df = pd.DataFrame([features], columns=heatwave_features)
#     X_scaled = heatwave_scaler.transform(X_df)
#     prob = float(heatwave_model.predict_proba(X_scaled)[0][1])
#     pred_default = int(prob >= 0.5)
#     pred_custom = int(prob >= 0.30)

#     return {
#         # "probability": round(prob, 4),
#         "probability__percent": round(prob * 100, 2),
#         # "prediction_default_0_5": pred_default,
#         # "prediction_custom_0_30": pred_custom
#     }

# def predict_landslide(features):
#     X_df = pd.DataFrame([features], columns=landslide_features)
#     X_scaled = landslide_scaler.transform(X_df)
#     prob = float(landslide_model.predict_proba(X_scaled)[0][1])
#     pred_default = int(prob >= 0.5)
#     pred_custom = int(prob >= 0.30)

#     return {
#         # "probability": round(prob, 4),
#         "probability__percent": round(prob * 100, 2),
#         # "prediction_default_0_5": pred_default,
#         # "prediction_custom_0_30": pred_custom
#     }

# # -------------------------
# # ROUTES
# # -------------------------
# @app.get("/")
# def home():
#     return {"status": "Multi Disaster Predictor API ✅ (Heatwave + Landslide)"}

# @app.get("/predict-live")
# def predict_live(lat: float, lon: float):
#     # fetch APIs
#     weather = fetch_weather(lat, lon)
    
#     elevation = fetch_elevation(lat, lon)

#     # heatwave
#     heatwave_features_dict = build_heatwave_features(weather)
#     heatwave_result = predict_heatwave(heatwave_features_dict)

#     # landslide (Elevation real + others dummy for now)
#     landslide_input = {
#         "Elevation": elevation,
#         "Slope": 25.0,
#         "Curvature": 3.0,
#         "Precipitation": float(weather["hourly"]["rain"][0]) if "rain" in weather["hourly"] else 0.0,
#         "Earthquake": 2.0,
#         "temperature": float(weather["daily"]["temperature_2m_max"][0]),
#         "moisture": 40.0,
#         "NDVI": 0.4,
#         "NDWI": 0.3,
#         "Lithology": 2.0
#     }

#     landslide_result = predict_landslide(landslide_input)

#     return {
#     "lat": lat,
#     "lon": lon,

#     "heatwave": heatwave_result,
#     "heatwave_features_used": heatwave_features_dict,

#     "landslide": landslide_result,
#     "landslide_features_used": landslide_input
#     }




from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# LOAD MODELS
# -------------------------
# Heatwave
heatwave_model = joblib.load("models/heatwave/best_heatwave_model.pkl")
heatwave_scaler = joblib.load("models/heatwave/scaler.pkl")
heatwave_features = joblib.load("models/heatwave/feature_columns.pkl")

# Landslide
landslide_model = joblib.load("models/landslide/landslide_model.pkl")
landslide_scaler = joblib.load("models/landslide/landslide_scaler.pkl")
landslide_features = joblib.load("models/landslide/landslide_feature_columns.pkl")

## Flood
flood_model = joblib.load("models/flood/flood_model.pkl")
flood_features = joblib.load("models/flood/flood_feature_columns.pkl")


# -------------------------
# APIs
# -------------------------
def fetch_weather(lat, lon):
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        "&daily=temperature_2m_min,temperature_2m_max,relative_humidity_2m_min,relative_humidity_2m_max,uv_index_max,shortwave_radiation_sum"
        "&hourly=dew_point_2m,pressure_msl,rain"
        "&current=rain,relative_humidity_2m,cloud_cover,wind_speed_10m"
    )
    return requests.get(url, timeout=20).json()

def fetch_elevation(lat, lon):
    url = f"https://api.opentopodata.org/v1/srtm90m?locations={lat},{lon}"
    data = requests.get(url, timeout=20).json()
    return float(data["results"][0]["elevation"])

# -------------------------
# SOIL
# -------------------------
def get_soil_type_id(sand, silt, clay, soc=0):
    """
    Soil type IDs:
    0 = Water / No Data
    1 = Peat
    2 = Silt
    3 = Clay
    4 = Loam
    5 = Sandy
    """
    if sand == 0 and silt == 0 and clay == 0:
        return 0

    if soc > 12:
        return 1
    if sand >= 70:
        return 5
    if clay >= 40:
        return 3
    if silt >= 80:
        return 2

    return 4

def fetch_soil(lat, lon):
    url = (
        f"https://rest.isric.org/soilgrids/v2.0/properties/query?"
        f"lat={lat}&lon={lon}&properties=sand,silt,clay,soc&depths=0-5cm&value=mean"
    )

    try:
        response = requests.get(url, timeout=15)
        data = response.json()
        layers = data.get("properties", {}).get("layers", [])

        if not layers:
            return {"soil_id": 0.0, "sand": 0.0, "silt": 0.0, "clay": 0.0, "soc": 0.0}

        stats = {}
        for layer in layers:
            name = layer["name"]
            # dg/kg -> %
            val = layer["depths"][0]["values"]["mean"] / 10
            stats[name] = float(val)

        sand = float(stats.get("sand", 0.0))
        silt = float(stats.get("silt", 0.0))
        clay = float(stats.get("clay", 0.0))

        # soc gets extra conversion
        soc_percent = float(stats.get("soc", 0.0) / 10)

        soil_id = float(get_soil_type_id(sand, silt, clay, soc_percent))

        return {
            "soil_id": soil_id,
            "sand": sand,
            "silt": silt,
            "clay": clay,
            "soc": soc_percent
        }

    except Exception:
        return {"soil_id": 0.0, "sand": 0.0, "silt": 0.0, "clay": 0.0, "soc": 0.0}
    
from datetime import datetime, timedelta
def fetch_earthquake_max_magnitude(lat, lon, radius_km=200, days=7):
    """
    Gets maximum earthquake magnitude near location in last N days.
    Uses USGS Earthquake API (no key).
    """
    try:
        starttime = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        url = (
            "https://earthquake.usgs.gov/fdsnws/event/1/query"
            f"?format=geojson&latitude={lat}&longitude={lon}"
            f"&maxradiuskm={radius_km}&starttime={starttime}"
        )

        data = requests.get(url, timeout=20).json()
        features = data.get("features", [])

        if not features:
            return 0.0  # no earthquake found

        mags = []
        for f in features:
            mag = f.get("properties", {}).get("mag", None)
            if mag is not None:
                mags.append(float(mag))

        return max(mags) if mags else 0.0

    except Exception:
        return 0.0

# -------------------------
# FEATURE BUILDERS
# -------------------------
def build_heatwave_features(w):
    min_temp = float(w["daily"]["temperature_2m_min"][0])
    max_temp = float(w["daily"]["temperature_2m_max"][0])
    min_hum = float(w["daily"]["relative_humidity_2m_min"][0])
    max_hum = float(w["daily"]["relative_humidity_2m_max"][0])

    uv_index = float(w["daily"]["uv_index_max"][0])
    solar_radiation = float(w["daily"]["shortwave_radiation_sum"][0])

    cloud_cover = float(w["current"]["cloud_cover"])
    wind_speed_kmh = float(w["current"]["wind_speed_10m"])

    pressure = float(w["hourly"]["pressure_msl"][0])
    dew_point = float(w["hourly"]["dew_point_2m"][0])

    # km/h -> m/s
    wind_speed = wind_speed_kmh / 3.6

    # derived
    apparent_temp = max_temp + (0.1 * dew_point)
    solar_impact = solar_radiation * (1 - cloud_cover / 100)

    temp_humidity_interaction = max_temp * max_hum
    heat_dome_proxy = max_temp * (1015 - pressure)
    temp_uv_interaction = max_temp * uv_index
    heat_stress = apparent_temp * uv_index

    return {
        "wind_speed": wind_speed,
        "cloud_cover": cloud_cover,
        "pressure_surface_level": pressure,
        "dew_point": dew_point,
        "uv_index": uv_index,
        "solar_radiation": solar_radiation,
        "max_temperature": max_temp,
        "min_temperature": min_temp,
        "max_humidity": max_hum,
        "min_humidity": min_hum,
        "apparent_temp": apparent_temp,
        "solar_impact": solar_impact,
        "temp_humidity_interaction": temp_humidity_interaction,
        "heat_dome_proxy": heat_dome_proxy,
        "temp_uv_interaction": temp_uv_interaction,
        "heat_stress": heat_stress
    }

# -------------------------
# PREDICT
# -------------------------
def predict_heatwave(features):
    X_df = pd.DataFrame([features], columns=heatwave_features)
    X_scaled = heatwave_scaler.transform(X_df)
    prob = float(heatwave_model.predict_proba(X_scaled)[0][1])

    return {
        "probability__percent": round(prob * 100, 2),
    }

def predict_landslide(features):
    X_df = pd.DataFrame([features], columns=landslide_features)
    X_scaled = landslide_scaler.transform(X_df)
    prob = float(landslide_model.predict_proba(X_scaled)[0][1])

    return {
        "probability__percent": round(prob * 100, 2),
    }
def predict_flood(features):
    X_df = pd.DataFrame([features], columns=flood_features)

    prob = float(flood_model.predict_proba(X_df)[0][1])

    return {
        "probability__percent": round(prob * 100, 2)
    }

# -------------------------
# ROUTES
# -------------------------
@app.get("/")
def home():
    return {"status": "Multi Disaster Predictor API ✅ (Heatwave + Landslide)"}

@app.get("/predict-live")
def predict_live(lat: float, lon: float):
    # fetch APIs
    weather = fetch_weather(lat, lon)
    elevation = fetch_elevation(lat, lon)

    # ✅ Soil API
    soil = fetch_soil(lat, lon)
    soil_id = soil["soil_id"]

    # heatwave
    heatwave_features_dict = build_heatwave_features(weather)
    heatwave_result = predict_heatwave(heatwave_features_dict)
    #EARTHQUACT
    eq_mag = fetch_earthquake_max_magnitude(lat, lon, radius_km=200, days=7)

    # moisture from weather current humidity if available
    moisture = float(weather["current"]["relative_humidity_2m"]) if "relative_humidity_2m" in weather["current"] else 40.0

    # landslide
    landslide_input = {
        "Elevation": elevation,
        "Slope": 25.0,
        "Curvature": 3.0,
        "Precipitation": float(weather["hourly"]["rain"][0]) if "rain" in weather["hourly"] else 0.0,
        "Earthquake": eq_mag,
        "temperature": float(weather["daily"]["temperature_2m_max"][0]),
        "moisture": moisture,
        "NDVI": 0.4,
        "NDWI": 0.3,
        "Lithology": float(soil_id)   # ✅ replaced dummy with soil id
    }
    
    flood_input = {
        "Rainfall (mm)": float(weather["current"]["rain"]) if "rain" in weather["current"] else 0.0,
        "Temperature (°C)": float(weather["daily"]["temperature_2m_max"][0]),
        "Humidity (%)": float(weather["current"]["relative_humidity_2m"]) if "relative_humidity_2m" in weather["current"] else 50.0,
        "River Discharge (m³/s)": 1500.0,   # dummy
        "Water Level (m)": 5.0,             # dummy
        "Elevation (m)": float(elevation),
        "Land Cover": "Urban",              # dummy
        "Soil Type": "Clay",                # from soil id optional (for now dummy)
        "Population Density": 5000.0,       # dummy
        "Infrastructure": 1                 # dummy (0/1)
    }
    flood_result = predict_flood(flood_input)


    landslide_result = predict_landslide(landslide_input)

    return {
        "lat": lat,
        "lon": lon,

        "heatwave": heatwave_result,
        "heatwave_features_used": heatwave_features_dict,

        "landslide": landslide_result,
        "landslide_features_used": landslide_input,

        "flood": flood_result,
        "flood_features_used": flood_input,


        # optional debug
        "soil_data": soil
    }
