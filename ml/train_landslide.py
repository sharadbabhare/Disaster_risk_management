import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ✅ load dataset
CSV_PATH = "ml/data/landslide_final.csv"
df = pd.read_csv(CSV_PATH)

# ✅ drop unwanted columns
drop_cols = ["Aspect", "Plan", "Profile"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# ✅ target
TARGET = "Landslide"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# model
model = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train_scaled, y_train)

# eval
y_pred = model.predict(X_test_scaled)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ✅ Save inside backend/models/landslide/
joblib.dump(model, "backend/models/landslide/landslide_model.pkl")
joblib.dump(scaler, "backend/models/landslide/landslide_scaler.pkl")
joblib.dump(list(X.columns), "backend/models/landslide/landslide_feature_columns.pkl")

print("✅ Saved landslide model files in backend/models/landslide/")
print("Features:", list(X.columns))
