import streamlit as st
import numpy as np
import joblib
import json
import os
import logging
import time

# ===== Logging =====
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
    page_title="ระบบประเมินความเสี่ยงโรคเบาหวาน",
    page_icon="🩺",
    layout="centered"
)

# ===== โหลดโมเดล =====
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

# ===== predict =====
def predict_with_logging(pipeline, input_data, user_id="anonymous"):
    start_time = time.time()

    prediction = pipeline.predict(input_data)[0]
    probabilities = pipeline.predict_proba(input_data)[0]

    elapsed = time.time() - start_time

    logger.info(
        f"result={prediction} prob={probabilities[1]:.4f} time={elapsed:.3f}s user={user_id}"
    )

    return prediction, probabilities

# ===== โหลด =====
with st.spinner("กำลังโหลดโมเดล..."):
    pipeline, feature_names, metadata = load_model()

# ===== Sidebar =====
with st.sidebar:
    st.header("ℹ️ เกี่ยวกับโมเดลนี้")
    st.write(f"ประเภทโมเดล: {metadata['model_type']}")
    st.write(f"ความแม่นยำ: {metadata['accuracy']*100:.1f}%")
    st.write(f"ข้อมูล train: {metadata['training_samples']} ราย")

    st.markdown("---")

    st.warning("""
⚠️ ข้อควรระวัง  
ผลลัพธ์นี้เป็นการประเมินเบื้องต้นจาก AI เท่านั้น  
ไม่สามารถใช้แทนการวินิจฉัยของแพทย์ได้  
กรุณาปรึกษาแพทย์หากมีข้อสงสัย
""")

# ===== Title =====
st.markdown("""
<h1 style='text-align:center;'>🩺 ระบบประเมินความเสี่ยงโรคเบาหวาน</h1>
<p style='text-align:center;'>
กรอกข้อมูลสุขภาพของคุณด้านล่าง ระบบจะประเมินความเสี่ยงการเป็นโรคเบาหวาน
โดยใช้โมเดล Machine Learning ที่ train จากข้อมูลผู้ป่วย 768 ราย
</p>
<hr>
""", unsafe_allow_html=True)

# ===== หัวข้อ =====
st.markdown("## 📋 กรอกข้อมูลสุขภาพ")

# ===== Input =====
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("จำนวนครั้งที่ตั้งครรภ์", 0, 20, 1)
    glucose = st.number_input("ระดับน้ำตาลในเลือด (mg/dL)", 0, 300, 120)
    blood_pressure = st.number_input("ความดันโลหิต Diastolic (mmHg)", 0, 150, 72)
    skin_thickness = st.number_input("ความหนาผิวหนัง Tricep (mm)", 0, 100, 20)

with col2:
    insulin = st.number_input("ระดับ Insulin (mu U/ml)", 0, 900, 80)
    bmi = st.number_input("ดัชนีมวลกาย BMI (kg/m²)", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.35)
    age = st.number_input("อายุ (ปี)", 1, 120, 30)

# ===== ปุ่มแดง =====
st.markdown("""
<style>
div.stButton > button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# ===== Predict =====
if st.button("🔍 ประเมินความเสี่ยง"):

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

    input_array = np.array([[input_dict[f] for f in feature_names]])

    user_id = f"user_{int(time.time())}"

    with st.spinner("กำลังทำนาย..."):
        prediction, probabilities = predict_with_logging(
            pipeline,
            input_array,
            user_id
        )

    prob = probabilities[1]

    st.markdown("---")

    if prediction == 1:
        st.error(f"ความเสี่ยงสูง ({prob*100:.2f}%)")
    else:
        st.success(f"ความเสี่ยงต่ำ ({(1-prob)*100:.2f}%)")

    st.progress(float(prob))