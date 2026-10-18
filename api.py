import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="YO(U)-terus PCOS Risk Assessment API",
    description="Backend API for PCOS Risk Prediction using Machine Learning",
    version="1.0.0"
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler
model_path = Path('pcos_model.pkl')
scaler_path = Path('scaler.pkl')

if not model_path.exists():
    raise FileNotFoundError(f"Model file not found at {model_path.resolve()}")
if not scaler_path.exists():
    raise FileNotFoundError(f"Scaler file not found at {scaler_path.resolve()}")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
feature_names = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else []

BLOOD_GROUP_MAP = {
    'A+': 11, 'A-': 12,
    'B+': 13, 'B-': 14,
    'AB+': 15, 'AB-': 16,
    'O+': 17, 'O-': 18
}

def to_binary(val):
    return 1.0 if str(val).strip().lower() in ['yes', 'y', '1', 'true'] else 0.0

def preprocess_request_data(data: dict) -> pd.DataFrame:
    """Preprocess raw user inputs into the exact 41 feature DataFrame expected by pcos_model.pkl"""
    age = float(data.get('age', 25))
    weight = float(data.get('weight', 60))
    height = float(data.get('height', 160))
    height_m = height / 100.0 if height > 0 else 1.6
    bmi = round(weight / (height_m ** 2), 2)
    bg_str = data.get('blood_group', 'B+')

    cycle_reg = data.get('cycle_regular', 'Regular')
    fsh = float(data.get('fsh', 4.5))
    lh = float(data.get('lh', 4.5))
    hip = float(data.get('hip', 36))
    waist = float(data.get('waist', 30))

    processed = {
        'Age (yrs)': age,
        'Weight (Kg)': weight,
        'Height(Cm)': height,
        'BMI': bmi,
        'Blood Group': float(BLOOD_GROUP_MAP.get(bg_str, 13)),
        'Pulse rate(bpm)': float(data.get('pulse', 72)),
        'RR (breaths/min)': float(data.get('respiration', 18)),
        'Hb(g/dl)': float(data.get('hb', 12.5)),
        'Cycle(R/I)': 2.0 if cycle_reg == 'Regular' else 4.0,
        'Cycle length(days)': float(data.get('cycle_length', 28)),
        'Marraige Status (Yrs)': float(data.get('marriage_status', 0)),
        'Pregnant(Y/N)': to_binary(data.get('pregnant', 'No')),
        'No. of aborptions': float(data.get('abortions', 0)),
        'I   beta-HCG(mIU/mL)': float(data.get('beta_hcg_1', 1.99)),
        'II    beta-HCG(mIU/mL)': float(data.get('beta_hcg_2', 1.99)),
        'FSH(mIU/mL)': fsh,
        'LH(mIU/mL)': lh,
        'FSH/LH': round(fsh / lh, 2) if lh > 0 else 1.0,
        'Hip(inch)': hip,
        'Waist(inch)': waist,
        'Waist:Hip Ratio': round(waist / hip, 2) if hip > 0 else 0.83,
        'TSH (mIU/L)': float(data.get('tsh', 2.5)),
        'AMH(ng/mL)': float(data.get('amh', 3.5)),
        'PRL(ng/mL)': float(data.get('prl', 15.0)),
        'Vit D3 (ng/mL)': float(data.get('vitamin_d', 30.0)),
        'PRG(ng/mL)': float(data.get('progesterone', 0.5)),
        'RBS(mg/dl)': float(data.get('rbs', 95.0)),
        'Weight gain(Y/N)': to_binary(data.get('weight_gain', 'No')),
        'hair growth(Y/N)': to_binary(data.get('hair_growth', 'No')),
        'Skin darkening (Y/N)': to_binary(data.get('skin_darkening', 'No')),
        'Hair loss(Y/N)': to_binary(data.get('hair_loss', 'No')),
        'Pimples(Y/N)': to_binary(data.get('pimples', 'No')),
        'Fast food (Y/N)': to_binary(data.get('fast_food', 'No')),
        'Reg.Exercise(Y/N)': to_binary(data.get('exercise', 'Yes')),
        'BP _Systolic (mmHg)': float(data.get('bp_systolic', 120)),
        'BP _Diastolic (mmHg)': float(data.get('bp_diastolic', 80)),
        'Follicle No. (L)': float(data.get('follicle_left', 6)),
        'Follicle No. (R)': float(data.get('follicle_right', 6)),
        'Avg. F size (L) (mm)': float(data.get('avg_follicle_left', 10)),
        'Avg. F size (R) (mm)': float(data.get('avg_follicle_right', 10)),
        'Endometrium (mm)': float(data.get('endometrium', 8))
    }

    df = pd.DataFrame([processed])
    if feature_names:
        df = df[feature_names]
    return df

def run_prediction_internal(input_data: dict):
    df = preprocess_request_data(input_data)
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

    importances = []
    if hasattr(model, 'feature_importances_'):
        imp_pairs = sorted(zip(feature_names, model.feature_importances_), key=lambda x: x[1], reverse=True)
        importances = [{'feature': name.strip(), 'importance': round(float(imp) * 100, 2)} for name, imp in imp_pairs[:6]]

    return {
        'prediction': 'Positive' if pred == 1 else 'Negative',
        'prediction_class': int(pred),
        'risk_level': risk_level,
        'confidence': round(confidence, 1),
        'pos_prob': round(pos_prob, 1),
        'neg_prob': round(neg_prob, 1),
        'importances': importances
    }

class PredictionRequest(BaseModel):
    age: Optional[float] = 25
    weight: Optional[float] = 60
    height: Optional[float] = 160
    blood_group: Optional[str] = 'B+'
    pulse: Optional[float] = 72
    respiration: Optional[float] = 18
    hb: Optional[float] = 12.5
    bp_systolic: Optional[float] = 120
    bp_diastolic: Optional[float] = 80
    cycle_regular: Optional[str] = 'Regular'
    cycle_length: Optional[float] = 28
    marriage_status: Optional[float] = 0
    pregnant: Optional[str] = 'No'
    abortions: Optional[float] = 0
    weight_gain: Optional[str] = 'No'
    hair_growth: Optional[str] = 'No'
    skin_darkening: Optional[str] = 'No'
    hair_loss: Optional[str] = 'No'
    pimples: Optional[str] = 'No'
    fast_food: Optional[str] = 'No'
    exercise: Optional[str] = 'Yes'
    amh: Optional[float] = 3.5
    fsh: Optional[float] = 4.5
    lh: Optional[float] = 4.5
    tsh: Optional[float] = 2.5
    prl: Optional[float] = 15.0
    vitamin_d: Optional[float] = 30.0
    progesterone: Optional[float] = 0.5
    rbs: Optional[float] = 95.0
    beta_hcg_1: Optional[float] = 1.99
    beta_hcg_2: Optional[float] = 1.99
    follicle_left: Optional[float] = 6
    follicle_right: Optional[float] = 6
    avg_follicle_left: Optional[float] = 10.0
    avg_follicle_right: Optional[float] = 10.0
    endometrium: Optional[float] = 8.0
    waist: Optional[float] = 30.0
    hip: Optional[float] = 36.0

@app.get("/")
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "PCOS Risk Assessment API",
        "model_loaded": True
    }

@app.post("/api/predict")
def predict_pcos(data: PredictionRequest):
    try:
        input_dict = data.model_dump()
        result = run_prediction_internal(input_dict)
        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
