import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import json

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Blood Diagnosis", layout="wide")

st.title("🩸 AI Blood Cell Diagnosis System")
st.markdown("Upload blood cell data to get **AI prediction + explanation**")

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    anomaly_model = joblib.load("models/classifier.pkl")
    disease_model = joblib.load("models/disease_model.pkl")

    with open("models/disease_columns.json", "r") as f:
        disease_columns = json.load(f)

    return anomaly_model, disease_model, disease_columns


model, disease_model, disease_columns = load_models()

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # -----------------------------
    # PREPARE DATA
    # -----------------------------
    X = df.drop(columns=["anomaly_label"], errors="ignore")

    # -----------------------------
    # SELECT ROW
    # -----------------------------
    index = st.slider("🔍 Select Cell Index", 0, len(X) - 1, 0)
    sample = X.iloc[[index]]

    # -----------------------------
    # ANOMALY PREDICTION
    # -----------------------------
    prediction = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0][1]

    st.subheader("🧪 Anomaly Detection")

    if prediction == 1:
        st.error(f"⚠️ Anomaly Detected (Confidence: {prob:.2f})")
    else:
        st.success(f"✅ Normal Cell (Confidence: {1 - prob:.2f})")

    # -----------------------------
    # DISEASE PREDICTION
    # -----------------------------
    sample_encoded = pd.get_dummies(sample)

    for col in disease_columns:
        if col not in sample_encoded:
            sample_encoded[col] = 0

    sample_encoded = sample_encoded[disease_columns]

    disease_prediction = disease_model.predict(sample_encoded)[0]

    st.subheader("🧬 Disease Prediction")
    st.info(f"Predicted Condition: {disease_prediction}")

    # -----------------------------
    # SHAP EXPLANATION
    # -----------------------------
    if prediction == 1:
        st.subheader("🧠 Why this cell is abnormal")
    else:
        st.subheader("🧠 Why this cell is normal")

    explainer = shap.Explainer(model, X)
    shap_values = explainer(sample)

    feature_names = X.columns

    # Correct SHAP selection
    if prediction == 1:
        values = shap_values[0, :, 1].values
    else:
        values = shap_values[0, :, 0].values

    top_indices = abs(values).argsort()[-5:][::-1]

    # -----------------------------
    # CORRECT FEATURE MEANINGS
    # -----------------------------
    if prediction == 1:
        feature_meaning = {
            "cytodiffusion_anomaly_score": "strong abnormal cell behavior",
            "cytodiffusion_classification_confidence": "high confidence in abnormality",
            "lobularity_score": "irregular nucleus segmentation",
            "granularity_score": "abnormal granules in cytoplasm",
            "cell_diameter_um": "unusual cell size",
            "eccentricity": "distorted cell shape",
            "cytoplasm_ratio": "imbalanced nucleus-to-cytoplasm ratio",
            "nucleus_area_pct": "large nucleus (immature cells)",
            "chromatin_density": "immature chromatin structure",
        }
    else:
        feature_meaning = {
            "cytodiffusion_anomaly_score": "low anomaly score indicating normal behavior",
            "cytodiffusion_classification_confidence": "low likelihood of abnormality",
            "lobularity_score": "normal nucleus segmentation",
            "granularity_score": "normal cytoplasm granularity",
            "cell_diameter_um": "typical cell size",
            "eccentricity": "regular cell shape",
            "cytoplasm_ratio": "balanced nucleus-to-cytoplasm ratio",
            "nucleus_area_pct": "normal nucleus size",
            "chromatin_density": "mature chromatin structure",
        }

    explanations = []

    for i in top_indices:
        feature = feature_names[i]
        if feature in feature_meaning:
            explanations.append(feature_meaning[feature])

    for exp in explanations:
        st.write(f"- {exp}")

    # -----------------------------
    # FINAL INTERPRETATION
    # -----------------------------
    st.subheader("🩸 Clinical Interpretation")

    if prediction == 1:
        st.warning(
            "The cell shows multiple abnormal characteristics, suggesting a high likelihood of pathological condition."
        )
    else:
        st.info(
            "The cell appears structurally normal with no strong indicators of abnormality."
        )

    # -----------------------------
    # SHAP VISUALIZATION
    # -----------------------------
    st.subheader("📊 Feature Impact Visualization")

    fig, ax = plt.subplots()

    if prediction == 1:
        shap.plots.waterfall(shap_values[0, :, 1], show=False)
    else:
        shap.plots.waterfall(shap_values[0, :, 0], show=False)

    st.pyplot(fig)

else:
    st.info("👆 Upload cleaned_data.csv to begin analysis")