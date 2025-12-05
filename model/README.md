# Model Artifacts

This directory contains the trained model artifacts for the HCT survival prediction application.

## Files

- `model.pkl` - The complete XGBoost pipeline (preprocessing + model)
- `model_info.pkl` - Feature metadata and validation information

## Generating Model Artifacts

Run the export script to generate the model files:

```bash
python scripts/export_model.py
```

This script:
1. Generates synthetic HCT data (matching the notebook)
2. Trains an XGBoost classifier with preprocessing pipeline
3. Saves the complete pipeline as `model.pkl`
4. Saves feature information as `model_info.pkl`

## Model Details

- **Algorithm**: XGBoost Classifier
- **Training Data**: 3000 synthetic patients
- **Features**: 9 patient characteristics
- **Preprocessing**: Missing value imputation, one-hot encoding, standardization
- **Random State**: 123 (for reproducibility)

## Using Your Own Model

To use a model trained from the original notebook:

1. Run the notebook to train your model
2. Save the pipeline using joblib:
   ```python
   import joblib
   joblib.dump(pipeline, 'model/model.pkl')
   ```
3. Update the feature information if needed

## Model Validation

The exported model should:
- Accept the 9 input features
- Return probabilities between 0 and 1
- Handle missing values automatically
- Be consistent across predictions