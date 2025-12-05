"""
Unit tests for the HCT prediction API
"""
import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app', 'backend'))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

@pytest.fixture
def valid_patient_data():
    """Valid patient data for testing"""
    return {
        "age": 45,
        "gender": "Female",
        "donor_type": "Matched sibling",
        "comorbidity_score": 1,
        "disease_type": "AML",
        "conditioning_intensity": "Reduced-intensity",
        "prior_transplants": 0,
        "time_from_diagnosis_days": 180,
        "treatment_days": 30
    }

def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"

def test_metrics_endpoint():
    """Test metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "prediction_count" in data
    assert "error_count" in data

def test_valid_prediction(valid_patient_data):
    """Test prediction with valid data"""
    response = client.post("/predict", json=valid_patient_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "probability" in data
    assert "prediction" in data
    assert "explainability" in data
    assert "disclaimer" in data
    
    # Check probability is between 0 and 1
    assert 0 <= data["probability"] <= 1
    
    # Check prediction is valid
    assert data["prediction"] in ["Likely to survive", "At risk"]

def test_invalid_age():
    """Test prediction with invalid age"""
    invalid_data = {
        "age": 15,  # Too young
        "gender": "Female",
        "donor_type": "Matched sibling",
        "comorbidity_score": 1,
        "disease_type": "AML",
        "conditioning_intensity": "Reduced-intensity",
        "prior_transplants": 0,
        "time_from_diagnosis_days": 180,
        "treatment_days": 30
    }
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422

def test_invalid_gender():
    """Test prediction with invalid gender"""
    invalid_data = {
        "age": 45,
        "gender": "Other",  # Invalid gender
        "donor_type": "Matched sibling",
        "comorbidity_score": 1,
        "disease_type": "AML",
        "conditioning_intensity": "Reduced-intensity",
        "prior_transplants": 0,
        "time_from_diagnosis_days": 180,
        "treatment_days": 30
    }
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422

def test_invalid_donor_type():
    """Test prediction with invalid donor type"""
    invalid_data = {
        "age": 45,
        "gender": "Female",
        "donor_type": "Invalid donor",  # Invalid donor type
        "comorbidity_score": 1,
        "disease_type": "AML",
        "conditioning_intensity": "Reduced-intensity",
        "prior_transplants": 0,
        "time_from_diagnosis_days": 180,
        "treatment_days": 30
    }
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422

def test_missing_fields():
    """Test prediction with missing required fields"""
    incomplete_data = {
        "age": 45,
        "gender": "Female"
        # Missing other required fields
    }
    response = client.post("/predict", json=incomplete_data)
    assert response.status_code == 422

def test_multiple_predictions(valid_patient_data):
    """Test multiple predictions to check consistency"""
    responses = []
    for _ in range(3):
        response = client.post("/predict", json=valid_patient_data)
        assert response.status_code == 200
        responses.append(response.json())
    
    # All predictions should be identical for same input
    first_prob = responses[0]["probability"]
    for response in responses[1:]:
        assert abs(response["probability"] - first_prob) < 1e-6