import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

CSV_PATH = "ml/data/flood_final.csv"
TARGET = "Flood Occurred"

df = pd.read_csv(CSV_PATH)

X = df.drop(columns=[TARGET])
y = df[TARGET]

# columns
cat_cols = ["Land Cover", "Soil Type"]
num_cols = [c for c in X.columns if c not in cat_cols]

# preprocessing
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

# model pipeline
model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("clf", RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced"))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ✅ save directly into backend folder
joblib.dump(model, "backend/models/flood/flood_model.pkl")
joblib.dump(list(X.columns), "backend/models/flood/flood_feature_columns.pkl")

print("✅ Flood model saved in backend/models/flood/")
