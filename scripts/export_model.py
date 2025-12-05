#!/usr/bin/env python3
"""
Export trained model from notebook to deployable artifact.
Reconstructs the XGBoost pipeline and saves it as model.pkl
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

def generate_synthetic_hct_data(n=2000, random_state=42):
    """Generate synthetic HCT data matching notebook structure"""
    rng = np.random.RandomState(random_state)
    age = rng.normal(loc=45, scale=12, size=n).clip(18, 80).astype(int)
    gender = rng.choice(['Male','Female'], size=n, p=[0.55,0.45])
    donor_type = rng.choice(['Matched sibling','Matched unrelated','Haploidentical','Cord blood'],
                            p=[0.35,0.4,0.2,0.05], size=n)
    comorbidity_score = rng.poisson(lam=2.0, size=n)
    disease_type = rng.choice(['AML','ALL','MDS','Lymphoma','Other'],
                              p=[0.25,0.15,0.2,0.25,0.15], size=n)
    conditioning_intensity = rng.choice(['Myeloablative','Reduced-intensity'], p=[0.4,0.6], size=n)
    prior_transplants = rng.binomial(1, 0.12, size=n)
    time_from_diagnosis_days = rng.exponential(scale=365, size=n).astype(int).clip(10, 5000)
    treatment_days = (30 + (comorbidity_score * 5) + rng.normal(0,10,size=n)).astype(int).clip(7, 365)

    risk = (
        0.03 * (age - 40) +
        0.4 * (comorbidity_score) +
        0.6 * (prior_transplants) +
        (donor_type == 'Matched sibling') * -0.6 +
        (donor_type == 'Haploidentical') * 0.4 +
        (conditioning_intensity == 'Myeloablative') * 0.3 +
        rng.normal(0, 0.8, size=n)
    )
    prob_survival = 1 / (1 + np.exp(risk))
    survival = (rng.rand(n) < prob_survival).astype(int)
    
    df = pd.DataFrame({
        'age': age,
        'gender': gender,
        'donor_type': donor_type,
        'comorbidity_score': comorbidity_score,
        'disease_type': disease_type,
        'conditioning_intensity': conditioning_intensity,
        'prior_transplants': prior_transplants,
        'time_from_diagnosis_days': time_from_diagnosis_days,
        'treatment_days': treatment_days,
        'survival': survival
    })
    
    # Add missing values
    for col in ['donor_type','comorbidity_score','conditioning_intensity']:
        mask = rng.rand(n) < 0.03
        df.loc[mask, col] = np.nan
    return df

def make_preprocessor(categorical_features, numeric_features):
    """Create preprocessing pipeline"""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    return preprocessor

def train_and_export_model(output_path='model/model.pkl', random_state=123):
    """Train model and export to deployable artifact"""
    print("Generating synthetic data...")
    df = generate_synthetic_hct_data(n=3000, random_state=random_state)
    
    target = 'survival'
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    categorical_features = ['gender','donor_type','disease_type','conditioning_intensity']
    numeric_features = [c for c in X.columns if c not in categorical_features]

    preprocessor = make_preprocessor(categorical_features, numeric_features)

    xgb_clf = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=random_state,
        n_jobs=2
    )

    pipeline = Pipeline(steps=[('preproc', preprocessor), ('xgb', xgb_clf)])

    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save model
    joblib.dump(pipeline, output_path)
    print(f"Model saved to: {output_path}")
    
    # Save feature info for validation
    feature_info = {
        'categorical_features': categorical_features,
        'numeric_features': numeric_features,
        'feature_names': list(X.columns)
    }
    info_path = output_path.replace('.pkl', '_info.pkl')
    joblib.dump(feature_info, info_path)
    print(f"Feature info saved to: {info_path}")
    
    # Test prediction
    test_pred = pipeline.predict_proba(X_test[:1])
    print(f"Test prediction probability: {test_pred[0][1]:.4f}")
    
    return pipeline, feature_info

if __name__ == "__main__":
    train_and_export_model()