"""
FastAPI backend for HCT survival prediction.
⚠️  FOR RESEARCH/DEMO USE ONLY. NOT FOR CLINICAL DECISION-MAKING.
"""
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('predictions.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="HCT Survival Prediction API",
    description="⚠️ FOR RESEARCH/DEMO USE ONLY. NOT FOR CLINICAL DECISION-MAKING.",
    version="1.0.0"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and metrics
model = None
feature_info = None
prediction_count = 0
error_count = 0

class PatientInput(BaseModel):
    """Input schema for patient data"""
    age: int = Field(..., ge=18, le=80, description="Patient age (18-80)")
    gender: str = Field(..., description="Patient gender")
    donor_type: str = Field(..., description="Donor type")
    comorbidity_score: int = Field(..., ge=0, le=10, description="Comorbidity score (0-10)")
    disease_type: str = Field(..., description="Disease type")
    conditioning_intensity: str = Field(..., description="Conditioning intensity")
    prior_transplants: int = Field(..., ge=0, le=1, description="Prior transplants (0 or 1)")
    time_from_diagnosis_days: int = Field(..., ge=1, le=5000, description="Days from diagnosis")
    treatment_days: int = Field(..., ge=1, le=365, description="Treatment duration in days")
    
    @validator('gender')
    def validate_gender(cls, v):
        if v not in ['Male', 'Female']:
            raise ValueError('Gender must be Male or Female')
        return v
    
    @validator('donor_type')
    def validate_donor_type(cls, v):
        valid_types = ['Matched sibling', 'Matched unrelated', 'Haploidentical', 'Cord blood']
        if v not in valid_types:
            raise ValueError(f'Donor type must be one of: {valid_types}')
        return v
    
    @validator('disease_type')
    def validate_disease_type(cls, v):
        valid_types = ['AML', 'ALL', 'MDS', 'Lymphoma', 'Other']
        if v not in valid_types:
            raise ValueError(f'Disease type must be one of: {valid_types}')
        return v
    
    @validator('conditioning_intensity')
    def validate_conditioning_intensity(cls, v):
        valid_types = ['Myeloablative', 'Reduced-intensity']
        if v not in valid_types:
            raise ValueError(f'Conditioning intensity must be one of: {valid_types}')
        return v

class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    probability: float = Field(..., description="Survival probability (0-1)")
    prediction: str = Field(..., description="Predicted outcome")
    explainability: Dict[str, Any] = Field(..., description="Feature importance and notes")
    disclaimer: str = Field(default="⚠️ FOR RESEARCH/DEMO USE ONLY. NOT FOR CLINICAL DECISION-MAKING.")

@app.on_event("startup")
async def load_model():
    """Load model and feature info on startup"""
    global model, feature_info
    try:
        # Get absolute path to model files
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.getenv('MODEL_PATH', os.path.join(base_dir, 'model', 'model.pkl'))
        info_path = model_path.replace('.pkl', '_info.pkl')
        
        model = joblib.load(model_path)
        feature_info = joblib.load(info_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def get_feature_importance(patient_data: pd.DataFrame) -> Dict[str, float]:
    """Get feature importance using XGBoost feature importance"""
    try:
        # Get feature importance from XGBoost model
        xgb_model = model.named_steps['xgb']
        feature_importance = xgb_model.feature_importances_
        
        # Transform input to get feature names after preprocessing
        preprocessed = model.named_steps['preproc'].transform(patient_data)
        
        # Get feature names from preprocessor
        feature_names = []
        for name, transformer, features in model.named_steps['preproc'].transformers_:
            if name == 'num':
                feature_names.extend(features)
            elif name == 'cat':
                if hasattr(transformer.named_steps['onehot'], 'get_feature_names_out'):
                    cat_features = transformer.named_steps['onehot'].get_feature_names_out(features)
                    feature_names.extend(cat_features)
        
        # Create importance dict
        importance_dict = {}
        for i, importance in enumerate(feature_importance):
            if i < len(feature_names):
                importance_dict[feature_names[i]] = float(importance)
        
        # Return top 5 most important features
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_importance[:5])
    
    except Exception as e:
        logger.warning(f"Could not compute feature importance: {e}")
        return {"note": "Feature importance unavailable"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/metrics")
async def get_metrics():
    """Get basic API metrics"""
    return {
        "prediction_count": prediction_count,
        "error_count": error_count,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("10/minute")
async def predict_survival(request: Request, patient: PatientInput):
    """Predict survival probability for a patient"""
    global prediction_count, error_count
    
    try:
        # Convert to DataFrame
        patient_data = pd.DataFrame([patient.dict()])
        
        # Make prediction
        probability = model.predict_proba(patient_data)[0, 1]
        prediction_class = "Likely to survive" if probability > 0.5 else "At risk"
        
        # Get feature importance
        feature_importance = get_feature_importance(patient_data)
        
        # Log prediction
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "patient_data": patient.dict(),
            "probability": float(probability),
            "prediction": prediction_class
        }
        logger.info(f"Prediction made: {json.dumps(log_data)}")
        
        prediction_count += 1
        
        return PredictionResponse(
            probability=float(probability),
            prediction=prediction_class,
            explainability={
                "feature_importance": feature_importance,
                "notes": "Higher scores indicate greater influence on survival prediction"
            }
        )
        
    except Exception as e:
        error_count += 1
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)