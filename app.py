import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="PCOS Risk Detection System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# MODEL & PREPROCESSING LOGIC
# ============================================================
BLOOD_GROUP_MAP = {
    'A+': 11, 'A-': 12,
    'B+': 13, 'B-': 14,
    'AB+': 15, 'AB-': 16,
    'O+': 17, 'O-': 18
}

@st.cache_resource
def load_model_and_scaler():
    """Load model and scaler with Streamlit caching"""
    model_path = Path('pcos_model.pkl')
    scaler_path = Path('scaler.pkl')

    if not model_path.exists():
        st.error(f"❌ Model file not found at {model_path.resolve()}")
        st.stop()
    if not scaler_path.exists():
        st.error(f"❌ Scaler file not found at {scaler_path.resolve()}")
        st.stop()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

try:
    model, scaler = load_model_and_scaler()
    feature_names = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else []
except Exception as e:
    st.error(f"Error initializing model: {str(e)}")
    st.stop()


def preprocess_inputs(data: dict) -> pd.DataFrame:
    """Preprocess raw user inputs into the exact 41 feature DataFrame expected by the model"""
    processed = {}

    # Demographics & Body Metrics
    age = float(data.get('age', 25))
    weight = float(data.get('weight', 60))
    height = float(data.get('height', 160))
    processed['Age (yrs)'] = age
    processed['Weight (Kg)'] = weight
    processed['Height(Cm)'] = height

    # Calculate BMI if needed
    height_m = height / 100.0 if height > 0 else 1.6
    processed['BMI'] = round(weight / (height_m ** 2), 2)

    # Blood Group
    bg_str = data.get('blood_group', 'B+')
    processed['Blood Group'] = float(BLOOD_GROUP_MAP.get(bg_str, 11))

    # Vitals & Blood Tests
    processed['Pulse rate(bpm)'] = float(data.get('pulse', 72))
    processed['RR (breaths/min)'] = float(data.get('respiration', 18))
    processed['Hb(g/dl)'] = float(data.get('hb', 12.5))

    # Menstrual Cycle & Reproductive History
    cycle_reg = data.get('cycle_regular', 'Regular')
    processed['Cycle(R/I)'] = 2.0 if cycle_reg == 'Regular' else 4.0
    processed['Cycle length(days)'] = float(data.get('cycle_length', 28))
    processed['Marraige Status (Yrs)'] = float(data.get('marriage_status', 0))

    def to_binary(val):
        return 1.0 if str(val).strip().lower() in ['yes', 'y', '1', 'true'] else 0.0

    processed['Pregnant(Y/N)'] = to_binary(data.get('pregnant', 'No'))
    processed['No. of aborptions'] = float(data.get('abortions', 0))

    # Hormonal Tests
    processed['I   beta-HCG(mIU/mL)'] = float(data.get('beta_hcg_1', 1.99))
    processed['II    beta-HCG(mIU/mL)'] = float(data.get('beta_hcg_2', 1.99))

    fsh = float(data.get('fsh', 4.5))
    lh = float(data.get('lh', 4.5))
    processed['FSH(mIU/mL)'] = fsh
    processed['LH(mIU/mL)'] = lh
    processed['FSH/LH'] = round(fsh / lh, 2) if lh > 0 else 1.0

    hip = float(data.get('hip', 36))
    waist = float(data.get('waist', 30))
    processed['Hip(inch)'] = hip
    processed['Waist(inch)'] = waist
    processed['Waist:Hip Ratio'] = round(waist / hip, 2) if hip > 0 else 0.83

    processed['TSH (mIU/L)'] = float(data.get('tsh', 2.5))
    processed['AMH(ng/mL)'] = float(data.get('amh', 3.5))
    processed['PRL(ng/mL)'] = float(data.get('prl', 15.0))
    processed['Vit D3 (ng/mL)'] = float(data.get('vitamin_d', 30.0))
    processed['PRG(ng/mL)'] = float(data.get('progesterone', 0.5))
    processed['RBS(mg/dl)'] = float(data.get('rbs', 95.0))

    # Lifestyle & Physical Symptoms
    processed['Weight gain(Y/N)'] = to_binary(data.get('weight_gain', 'No'))
    processed['hair growth(Y/N)'] = to_binary(data.get('hair_growth', 'No'))
    processed['Skin darkening (Y/N)'] = to_binary(data.get('skin_darkening', 'No'))
    processed['Hair loss(Y/N)'] = to_binary(data.get('hair_loss', 'No'))
    processed['Pimples(Y/N)'] = to_binary(data.get('pimples', 'No'))
    processed['Fast food (Y/N)'] = to_binary(data.get('fast_food', 'No'))
    processed['Reg.Exercise(Y/N)'] = to_binary(data.get('exercise', 'Yes'))

    # Blood Pressure
    processed['BP _Systolic (mmHg)'] = float(data.get('bp_systolic', 120))
    processed['BP _Diastolic (mmHg)'] = float(data.get('bp_diastolic', 80))

    # Ultrasound Findings
    processed['Follicle No. (L)'] = float(data.get('follicle_left', 6))
    processed['Follicle No. (R)'] = float(data.get('follicle_right', 6))
    processed['Avg. F size (L) (mm)'] = float(data.get('avg_follicle_left', 10))
    processed['Avg. F size (R) (mm)'] = float(data.get('avg_follicle_right', 10))
    processed['Endometrium (mm)'] = float(data.get('endometrium', 8))

    df = pd.DataFrame([processed])
    if feature_names:
        df = df[feature_names]
    return df


def run_prediction(input_data: dict):
    """Execute prediction pipeline"""
    df = preprocess_inputs(input_data)
    scaled_df = scaler.transform(df)

    pred = model.predict(scaled_df)[0]
    probs = model.predict_proba(scaled_df)[0]

    pos_prob = float(probs[1]) * 100
    neg_prob = float(probs[0]) * 100
    confidence = max(pos_prob, neg_prob)

    if pred == 1:
        risk_level = "High Risk" if pos_prob >= 75 else "Moderate Risk"
    else:
        risk_level = "Mild Risk" if pos_prob >= 35 else "Low Risk"

    # Top Feature Importances
    importances = []
    if hasattr(model, 'feature_importances_'):
        imp_pairs = sorted(zip(feature_names, model.feature_importances_), key=lambda x: x[1], reverse=True)
        importances = [{'feature': name.strip(), 'importance': round(imp * 100, 2)} for name, imp in imp_pairs[:6]]

    return {
        'prediction': 'Positive' if pred == 1 else 'Negative',
        'prediction_class': int(pred),
        'risk_level': risk_level,
        'confidence': round(confidence, 1),
        'pos_prob': round(pos_prob, 1),
        'neg_prob': round(neg_prob, 1),
        'importances': importances
    }


# ============================================================
# STREAMLIT UI APP
# ============================================================

# Header
st.title("🩺 PCOS Risk Detection Calculator")
st.markdown("Enter patient clinical, laboratory, and ultrasound metrics below for an instant machine learning risk assessment.")
st.divider()

# Sidebar Info
with st.sidebar:
    st.header("⚙️ App Controls")
    st.info("💡 **Model Info**: ExtraTrees Classifier trained on 41 clinical parameters.")

    if st.button("🔄 Reset Form", use_container_width=True):
        st.rerun()

    st.markdown("---")
    st.caption("Made for PCOS Risk Evaluation")

# Form Inputs
with st.form("pcos_form"):
    tab1, tab2, tab3, tab4 = st.tabs([
        "📌 1. Basic & Vitals",
        "🩸 2. Menstrual & Symptoms",
        "🧪 3. Hormones & Labs",
        "🩺 4. Ultrasound & Measurements"
    ])

    # Tab 1: Basic & Vitals
    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age (years)", min_value=12, max_value=60, value=25)
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=60.0, step=0.5)
            height = st.number_input("Height (cm)", min_value=100.0, max_value=220.0, value=160.0, step=0.5)
        with col2:
            blood_group = st.selectbox("Blood Group", ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'], index=2)
            pulse = st.number_input("Pulse Rate (bpm)", min_value=40, max_value=160, value=72)
            respiration = st.number_input("Respiration Rate (breaths/min)", min_value=10, max_value=40, value=18)
        with col3:
            hb = st.number_input("Hemoglobin Hb (g/dl)", min_value=5.0, max_value=20.0, value=12.5, step=0.1)
            bp_systolic = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=200, value=120)
            bp_diastolic = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=130, value=80)

    # Tab 2: Menstrual & Symptoms
    with tab2:
        col1, col2, col3 = st.columns(3)
        with col1:
            cycle_regular = st.selectbox("Menstrual Cycle Regularity", ["Regular", "Irregular"])
            cycle_length = st.number_input("Cycle Length (days)", min_value=15, max_value=60, value=28)
            marriage_status = st.number_input("Marriage Duration (Years)", min_value=0, max_value=40, value=0)
            pregnant = st.selectbox("Pregnant currently?", ["No", "Yes"])
            abortions = st.number_input("Number of Abortions", min_value=0, max_value=10, value=0)
        with col2:
            weight_gain = st.selectbox("Noticeable Weight Gain?", ["No", "Yes"])
            hair_growth = st.selectbox("Excess Facial/Body Hair (Hirsutism)?", ["No", "Yes"])
            skin_darkening = st.selectbox("Skin Darkening (Acanthosis)?", ["No", "Yes"])
        with col3:
            hair_loss = st.selectbox("Hair Loss / Thinning?", ["No", "Yes"])
            pimples = st.selectbox("Acne / Pimples Breakouts?", ["No", "Yes"])
            fast_food = st.selectbox("Frequent Fast Food Consumption?", ["No", "Yes"])
            exercise = st.selectbox("Regular Exercise Routine?", ["Yes", "No"])

    # Tab 3: Hormones & Labs
    with tab3:
        col1, col2, col3 = st.columns(3)
        with col1:
            amh = st.number_input("AMH (ng/mL)", min_value=0.0, max_value=50.0, value=3.5, step=0.1)
            fsh = st.number_input("FSH (mIU/mL)", min_value=0.0, max_value=50.0, value=4.5, step=0.1)
            lh = st.number_input("LH (mIU/mL)", min_value=0.0, max_value=50.0, value=4.5, step=0.1)
        with col2:
            tsh = st.number_input("TSH (mIU/L)", min_value=0.0, max_value=30.0, value=2.5, step=0.1)
            prl = st.number_input("Prolactin PRL (ng/mL)", min_value=0.0, max_value=100.0, value=15.0, step=0.1)
            vitamin_d = st.number_input("Vitamin D3 (ng/mL)", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
        with col3:
            progesterone = st.number_input("Progesterone PRG (ng/mL)", min_value=0.0, max_value=30.0, value=0.5, step=0.1)
            rbs = st.number_input("Random Blood Sugar RBS (mg/dl)", min_value=50, max_value=300, value=95)
            beta_hcg_1 = st.number_input("I Beta-HCG (mIU/mL)", min_value=0.0, max_value=500.0, value=1.99, step=0.1)
            beta_hcg_2 = st.number_input("II Beta-HCG (mIU/mL)", min_value=0.0, max_value=500.0, value=1.99, step=0.1)

    # Tab 4: Ultrasound & Measurements
    with tab4:
        col1, col2, col3 = st.columns(3)
        with col1:
            follicle_left = st.number_input("Follicle Count (Left Ovary)", min_value=0, max_value=30, value=6)
            follicle_right = st.number_input("Follicle Count (Right Ovary)", min_value=0, max_value=30, value=6)
        with col2:
            avg_follicle_left = st.number_input("Avg Follicle Size Left (mm)", min_value=0.0, max_value=30.0, value=10.0, step=0.5)
            avg_follicle_right = st.number_input("Avg Follicle Size Right (mm)", min_value=0.0, max_value=30.0, value=10.0, step=0.5)
        with col3:
            endometrium = st.number_input("Endometrium Thickness (mm)", min_value=0.0, max_value=25.0, value=8.0, step=0.5)
            waist = st.number_input("Waist Circumference (inches)", min_value=20.0, max_value=60.0, value=30.0, step=0.5)
            hip = st.number_input("Hip Circumference (inches)", min_value=20.0, max_value=70.0, value=36.0, step=0.5)

    st.markdown("---")
    submit_btn = st.form_submit_button("🔬 Predict PCOS Risk", use_container_width=True, type="primary")

# Prediction Execution & Results
if submit_btn:
    user_inputs = {
        'age': age, 'weight': weight, 'height': height, 'blood_group': blood_group,
        'pulse': pulse, 'respiration': respiration, 'hb': hb, 'bp_systolic': bp_systolic, 'bp_diastolic': bp_diastolic,
        'cycle_regular': cycle_regular, 'cycle_length': cycle_length, 'marriage_status': marriage_status,
        'pregnant': pregnant, 'abortions': abortions, 'weight_gain': weight_gain, 'hair_growth': hair_growth,
        'skin_darkening': skin_darkening, 'hair_loss': hair_loss, 'pimples': pimples, 'fast_food': fast_food,
        'exercise': exercise, 'amh': amh, 'fsh': fsh, 'lh': lh, 'tsh': tsh, 'prl': prl, 'vitamin_d': vitamin_d,
        'progesterone': progesterone, 'rbs': rbs, 'beta_hcg_1': beta_hcg_1, 'beta_hcg_2': beta_hcg_2,
        'follicle_left': follicle_left, 'follicle_right': follicle_right, 'avg_follicle_left': avg_follicle_left,
        'avg_follicle_right': avg_follicle_right, 'endometrium': endometrium, 'waist': waist, 'hip': hip
    }

    with st.spinner("🧠 Analyzing patient clinical parameters..."):
        res = run_prediction(user_inputs)

    st.subheader("📊 Assessment Result")

    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        if res['prediction_class'] == 1:
            st.error(f"### Result: {res['prediction']} (PCOS Detected)")
        else:
            st.success(f"### Result: {res['prediction']} (No PCOS)")

    with res_col2:
        st.metric("Risk Level", res['risk_level'])

    with res_col3:
        st.metric("Model Confidence", f"{res['confidence']}%")

    # Probabilities
    st.markdown("#### Probability Breakdown")
    st.progress(res['pos_prob'] / 100.0, text=f"PCOS Positive Probability: {res['pos_prob']}%")
    st.progress(res['neg_prob'] / 100.0, text=f"PCOS Negative Probability: {res['neg_prob']}%")

    # Top Features
    if res['importances']:
        st.markdown("#### Key Risk Factors (Feature Importance)")
        imp_df = pd.DataFrame(res['importances'])
        st.bar_chart(imp_df.set_index('feature')['importance'])

    # Recommendations
    st.markdown("#### Recommendations")
    if res['prediction_class'] == 1:
        st.warning("""
        - 🩺 Consult a gynecologist or endocrinologist for comprehensive clinical evaluation.
        - 🧪 Schedule a full pelvic ultrasound and hormonal blood panel (AMH, LH/FSH, Free Testosterone).
        - 🥗 Consider adopting an anti-inflammatory, low-GI diet to help balance insulin levels.
        - 🏃‍♀️ Engage in 150+ minutes of moderate physical activity weekly.
        """)
    else:
        st.info("""
        - 🍏 Maintain a balanced, nutrient-rich diet and regular exercise regimen.
        - 📅 Continue tracking your monthly menstrual cycle regularity.
        - 🩺 Keep up with annual routine health checkups.
        """)
