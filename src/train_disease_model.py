import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load original dataset (NOT cleaned)
df = pd.read_csv("data/blood_cell_anomaly_detection.csv")

# Encode target
y = df["disease_category"]

# Drop non-useful columns
X = df.drop(columns=["cell_id", "cell_type", "disease_category"])

# Convert categorical → numeric
X = pd.get_dummies(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

import json

# Save feature columns
with open("models/disease_columns.json", "w") as f:
    json.dump(list(X.columns), f)

import json

with open("models/disease_columns.json", "w") as f:
    json.dump(list(X.columns), f)

# Save
joblib.dump(model, "models/disease_model.pkl")

print("✅ Disease model trained")