import streamlit as st
import numpy as np
import joblib
import json
import os

# 🔥 Logging (1.7)
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("ml_service.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ===== ตั้งค่า =====
st.set_page_config(
    page_title="ระบบทำนายความเสี่ยงโรคเบาหวาน",
    page_icon="🩺",
    layout="centered"
)

# ===== โหลดโมเดล + feature =====
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(BASE_DIR, "model_artifacts", "diabetes_pipeline.pkl")
    feature_path = os.path.join(BASE_DIR, "model_artifacts", "feature_names.json")
    metadata_path = os.path.join(BASE_DIR, "model_artifacts", "model_metadata.json")

    pipeline = joblib.load(model_path)

    with open(feature_path, "r") as f:
        feature_names = json.load(f)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return pipeline, feature_names, metadata

# ===== Logging predict =====
def predict_with_logging(pipeline, input_data, user_id="anonymous"):
    start_time = time.time()

    try:
        prediction = pipeline.predict(input_data)[0]
        probabilities = pipeline.predict_proba(input_data)[0]

        elapsed = time.time() - start_time

        logger.info(
            f"result={prediction} prob={probabilities[1]:.4f} time={elapsed:.3f}s user={user_id}"
        )

        return prediction, probabilities

    except Exception as e:
        logger.error(f"error={str(e)} input={input_data.tolist()}")
        raise

# ===== โหลด =====
with st.spinner("กำลังโหลดโมเดล..."):
    pipeline, feature_names, metadata = load_model()

# ===== Sidebar =====
with st.sidebar:
    st.header("ℹ️ Model Info")
    st.write(f"Model: {metadata['model_type']}")
    st.write(f"Accuracy: {metadata['accuracy']*100:.2f}%")
    st.write(f"Train size: {metadata['training_samples']}")

# ===== UI =====
st.title("🩺 Diabetes Prediction")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose", 0, 300, 120)
    blood_pressure = st.number_input("BloodPressure", 0, 150, 72)
    skin_thickness = st.number_input("SkinThickness", 0, 100, 20)

with col2:
    insulin = st.number_input("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("DiabetesPedigreeFunction", 0.0, 3.0, 0.35)
    age = st.number_input("Age", 1, 120, 30)

# ===== Predict =====
if st.button("Predict"):

    # 🔥 map input ให้ตรงชื่อ feature
    input_dict = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }

    # 🔥 เรียง feature ให้ตรง model
    input_array = np.array([[input_dict[f] for f in feature_names]])

    user_id = f"user_{int(time.time())}"

    with st.spinner("กำลังทำนาย..."):
        prediction, probabilities = predict_with_logging(
            pipeline,
            input_array,
            user_id
        )

    prob = probabilities[1]

    st.subheader("Result")

    if prediction == 1:
        st.error(f"High Risk ({prob*100:.2f}%)")
    else:
        st.success(f"Low Risk ({(1-prob)*100:.2f}%)")

    st.progress(float(prob))