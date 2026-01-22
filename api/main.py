from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import uvicorn
import os
from datetime import datetime

# Import local modules
from preprocessing import SepsisDataPreprocessor
from models import SepsisPredictor

app = FastAPI(
    title="Sepsis Prediction API",
    description="API for predicting sepsis risk in ICU patients",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
predictor = None
preprocessor = SepsisDataPreprocessor()

# Data models
class PatientData(BaseModel):
    PRG: float
    PL: float  
    PR: float  
    SK: float  
    TS: float  
    M11: float 
    BD2: float 
    Age: float 
    Insurance: float

class PredictionRequest(BaseModel):
    patient_data: PatientData

class PredictionResponse(BaseModel):
    prediction_time: str
    sepsis_probability: float
    risk_level: str
    confidence: float
    clinical_indicators: List[str]
    immediate_actions: List[str]
    monitoring_recommendations: List[str]

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global predictor
    try:
        predictor = SepsisPredictor()
        if os.path.exists('models'):
            predictor.load_models('models')
            print("✓ Models loaded successfully")
        else:
            print("⚠ No pre-trained models found. Train models first.")
    except Exception as e:
        print(f"Error loading models: {e}")

@app.get("/")
async def root():
    return {
        "service": "Sepsis Prediction API",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "features": "/features"
        }
    }

@app.get("/health")
async def health_check():
    model_status = "loaded" if predictor and predictor.models else "not_loaded"
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": model_status
    }

@app.get("/features")
async def get_features():
    """Get feature information"""
    feature_info = {}
    for feature in preprocessor.feature_names:
        feature_info[feature] = {
            "description": preprocessor.get_feature_description(feature),
            "medical_importance": "Critical for sepsis assessment"
        }
    return feature_info

@app.post("/predict", response_model=PredictionResponse)
async def predict_sepsis(request: PredictionRequest):
    """Predict sepsis risk for a patient"""
    if not predictor or not predictor.models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Convert to DataFrame
        patient_dict = request.patient_data.dict()
        df = pd.DataFrame([patient_dict])
        
        # Preprocess
        X_processed, _, _ = preprocessor.preprocess_data(df, is_train=False)
        
        # Create sepsis features
        X_enhanced = preprocessor.create_sepsis_features(X_processed)
        
        # Make prediction
        probability = predictor.ensemble_predict(X_enhanced)[0]
        
        # Generate clinical analysis
        analysis = generate_clinical_analysis(patient_dict, probability)
        
        return PredictionResponse(
            prediction_time=datetime.now().isoformat(),
            sepsis_probability=round(float(probability), 4),
            risk_level=analysis['risk_level'],
            confidence=analysis['confidence'],
            clinical_indicators=analysis['clinical_indicators'],
            immediate_actions=analysis['immediate_actions'],
            monitoring_recommendations=analysis['monitoring_recommendations']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def generate_clinical_analysis(patient_data: dict, probability: float) -> dict:
    """Generate clinical analysis based on prediction"""
    
    # Get risk indicators
    clinical_indicators = preprocessor.get_sepsis_risk_interpretation(patient_data)
    
    # Determine risk level and actions
    if probability >= 0.8:
        risk_level = "CRITICAL"
        immediate_actions = [
            "🚨 ACTIVATE SEPSIS PROTOCOL",
            "Obtain blood cultures BEFORE antibiotics",
            "Administer broad-spectrum antibiotics within 1 HOUR",
            "Initiate fluid resuscitation (30ml/kg)",
            "Measure serum lactate STAT",
            "Notify ICU team immediately"
        ]
        monitoring = [
            "Continuous hemodynamic monitoring",
            "Hourly vital signs",
            "Repeat lactate in 2-4 hours"
        ]
        confidence = min(probability * 100 * 1.1, 95.0)
        
    elif probability >= 0.6:
        risk_level = "HIGH"
        immediate_actions = [
            "⚠️ High suspicion for sepsis",
            "Obtain blood cultures",
            "Prepare antibiotics",
            "Start IV access",
            "Check lactate level",
            "Notify physician"
        ]
        monitoring = [
            "Vital signs every 30 minutes",
            "Repeat assessment in 1 hour"
        ]
        confidence = probability * 100
        
    elif probability >= 0.4:
        risk_level = "MODERATE"
        immediate_actions = [
            "Monitor closely for sepsis signs",
            "Consider blood cultures",
            "Review medications"
        ]
        monitoring = [
            "Vital signs hourly",
            "Repeat assessment in 2-4 hours"
        ]
        confidence = probability * 100 * 0.9
        
    elif probability >= 0.2:
        risk_level = "LOW"
        immediate_actions = [
            "Continue routine monitoring",
            "Maintain vigilance"
        ]
        monitoring = [
            "Standard monitoring"
        ]
        confidence = probability * 100 * 0.8
        
    else:
        risk_level = "VERY LOW"
        immediate_actions = ["Routine clinical care"]
        monitoring = ["Scheduled assessments"]
        confidence = max(probability * 100 * 0.7, 10.0)
    
    return {
        'risk_level': risk_level,
        'confidence': round(confidence, 1),
        'clinical_indicators': clinical_indicators,
        'immediate_actions': immediate_actions,
        'monitoring_recommendations': monitoring
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)