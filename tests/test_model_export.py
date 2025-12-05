"""
Test model export functionality
"""
import os
import sys
import tempfile
import joblib
import pandas as pd

# Add scripts to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from export_model import generate_synthetic_hct_data, make_preprocessor, train_and_export_model

def test_generate_synthetic_data():
    """Test synthetic data generation"""
    df = generate_synthetic_hct_data(n=100, random_state=42)
    
    # Check shape
    assert df.shape[0] == 100
    assert df.shape[1] == 10  # 9 features + 1 target
    
    # Check columns
    expected_cols = [
        'age', 'gender', 'donor_type', 'comorbidity_score', 'disease_type',
        'conditioning_intensity', 'prior_transplants', 'time_from_diagnosis_days',
        'treatment_days', 'survival'
    ]
    assert list(df.columns) == expected_cols
    
    # Check data types and ranges
    assert df['age'].min() >= 18
    assert df['age'].max() <= 80
    assert df['survival'].isin([0, 1]).all()
    assert df['prior_transplants'].isin([0, 1]).all()

def test_preprocessor():
    """Test preprocessor creation"""
    categorical_features = ['gender', 'donor_type']
    numeric_features = ['age', 'comorbidity_score']
    
    preprocessor = make_preprocessor(categorical_features, numeric_features)
    
    # Test with sample data
    df = pd.DataFrame({
        'age': [45, 60],
        'gender': ['Male', 'Female'],
        'donor_type': ['Matched sibling', 'Matched unrelated'],
        'comorbidity_score': [1, 3]
    })
    
    transformed = preprocessor.fit_transform(df)
    assert transformed.shape[0] == 2
    assert transformed.shape[1] > 4  # Should be expanded due to one-hot encoding

def test_model_export():
    """Test complete model export process"""
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, 'test_model.pkl')
        
        # Export model
        pipeline, feature_info = train_and_export_model(
            output_path=model_path, 
            random_state=42
        )
        
        # Check files exist
        assert os.path.exists(model_path)
        assert os.path.exists(model_path.replace('.pkl', '_info.pkl'))
        
        # Load and test model
        loaded_pipeline = joblib.load(model_path)
        loaded_info = joblib.load(model_path.replace('.pkl', '_info.pkl'))
        
        # Test prediction
        test_data = pd.DataFrame([{
            'age': 45,
            'gender': 'Female',
            'donor_type': 'Matched sibling',
            'comorbidity_score': 1,
            'disease_type': 'AML',
            'conditioning_intensity': 'Reduced-intensity',
            'prior_transplants': 0,
            'time_from_diagnosis_days': 180,
            'treatment_days': 30
        }])
        
        prob = loaded_pipeline.predict_proba(test_data)[0, 1]
        assert 0 <= prob <= 1
        
        # Check feature info
        assert 'categorical_features' in loaded_info
        assert 'numeric_features' in loaded_info
        assert 'feature_names' in loaded_info