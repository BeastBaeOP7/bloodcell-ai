import shap
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("models/classifier.pkl")

# Load data
df = pd.read_csv("data/cleaned_data.csv")

X = df.drop(columns=["anomaly_label"])

# SHAP explainer
explainer = shap.Explainer(model, X)

# Pick one sample
sample = X.iloc[[0]]

shap_values = explainer(sample)
feature_names = X.columns
values = shap_values[0, :, 1].values

# Get top features
top_indices = abs(values).argsort()[-5:][::-1]

feature_meaning = {
    "cytodiffusion_anomaly_score": "overall abnormal cell behavior",
    "cytodiffusion_classification_confidence": "high AI confidence in abnormality",
    "lobularity_score": "irregular nucleus segmentation (linked to infections)",
    "granularity_score": "abnormal granules in cytoplasm (infection indicator)",
    "cell_diameter_um": "unusual cell size",
    "eccentricity": "elongated or distorted shape",
    "cytoplasm_ratio": "imbalance between nucleus and cytoplasm",
    "nucleus_area_pct": "large nucleus (common in immature/blast cells)",
    "chromatin_density": "immature or less compact chromatin",
}

print("\n🧠 AI Medical Explanation:\n")

for i in top_indices:
    feature = feature_names[i]
    if feature in feature_meaning:
        print(f"- {feature_meaning[feature]}")
# Print explanation
shap.plots.waterfall(shap_values[0, :, 1])
plt.show()