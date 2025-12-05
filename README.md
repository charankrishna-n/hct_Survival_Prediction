# HCT Survival Prediction App

⚠️ **FOR RESEARCH/DEMO USE ONLY. NOT FOR CLINICAL DECISION-MAKING.**

A production-ready machine learning application that predicts post-transplant survival probability for Hematopoietic Cell Transplantation (HCT) patients using XGBoost.

## Features

- **FastAPI Backend**: RESTful API with input validation, rate limiting, and logging
- **Streamlit Frontend**: Interactive web interface for predictions
- **Model Explainability**: Feature importance using XGBoost built-in importance
- **Production Ready**: Docker containerization, CI/CD pipeline, comprehensive testing
- **Security**: Input sanitization, rate limiting, audit logging
- **AWS Deployment**: Ready for ECS/Fargate deployment

## Project Structure

```
├── app/
│   ├── backend/           # FastAPI application
│   └── frontend/          # Streamlit application
├── model/                 # Model artifacts (generated)
├── scripts/               # Model export and utilities
├── tests/                 # Unit tests
├── .github/workflows/     # CI/CD pipeline
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container definition
├── docker-compose.yml    # Local development setup
└── README.md
```

## Quick Start

### 1. Export the Model

First, generate the model artifact from the notebook:

```bash
python scripts/export_model.py
```

This creates:
- `model/model.pkl` - The trained XGBoost pipeline
- `model/model_info.pkl` - Feature metadata

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run with Docker Compose (Recommended)

```bash
# Build and start both backend and frontend
docker-compose up --build

# Access the applications:
# - Backend API: http://localhost:8000
# - Frontend UI: http://localhost:8501
# - API docs: http://localhost:8000/docs
```

### 4. Run Manually (Development)

**Backend:**
```bash
cd app/backend
python main.py
# API available at http://localhost:8000
```

**Frontend:**
```bash
cd app/frontend
streamlit run streamlit_app.py
# UI available at http://localhost:8501
```

## API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Make Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "gender": "Female",
    "donor_type": "Matched sibling",
    "comorbidity_score": 1,
    "disease_type": "AML",
    "conditioning_intensity": "Reduced-intensity",
    "prior_transplants": 0,
    "time_from_diagnosis_days": 180,
    "treatment_days": 30
  }'
```

### Response Format
```json
{
  "probability": 0.73,
  "prediction": "Likely to survive",
  "explainability": {
    "feature_importance": {
      "age": 0.25,
      "comorbidity_score": 0.20,
      "donor_type_Matched sibling": 0.18
    },
    "notes": "Higher scores indicate greater influence on survival prediction"
  },
  "disclaimer": "⚠️ FOR RESEARCH/DEMO USE ONLY. NOT FOR CLINICAL DECISION-MAKING."
}
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=app/backend --cov-report=html
```

## Model Information

- **Algorithm**: XGBoost Classifier
- **Features**: 9 patient characteristics (age, gender, donor type, etc.)
- **Target**: Binary survival outcome
- **Preprocessing**: Automatic handling of missing values, one-hot encoding, standardization
- **Explainability**: Feature importance scores

### Input Features

| Feature | Type | Valid Values |
|---------|------|--------------|
| age | int | 18-80 |
| gender | str | Male, Female |
| donor_type | str | Matched sibling, Matched unrelated, Haploidentical, Cord blood |
| comorbidity_score | int | 0-10 |
| disease_type | str | AML, ALL, MDS, Lymphoma, Other |
| conditioning_intensity | str | Myeloablative, Reduced-intensity |
| prior_transplants | int | 0, 1 |
| time_from_diagnosis_days | int | 1-5000 |
| treatment_days | int | 1-365 |

## AWS Deployment

### Option 1: ECS Fargate

1. **Push to ECR:**
```bash
# Create ECR repository
aws ecr create-repository --repository-name hct-prediction

# Get login token
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build and push
docker build -t hct-prediction .
docker tag hct-prediction:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/hct-prediction:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/hct-prediction:latest
```

2. **Create ECS Service:**
   - Use the pushed image in ECS task definition
   - Configure ALB for load balancing
   - Set environment variables (MODEL_PATH, etc.)

### Option 2: Elastic Beanstalk

1. **Create application:**
```bash
eb init hct-prediction --platform docker
eb create production
```

2. **Deploy:**
```bash
eb deploy
```

## Security Considerations

### Current Security Features
- Input validation with Pydantic
- Rate limiting (10 requests/minute per IP)
- Request/response logging
- CORS configuration
- Health checks

### Production Security Recommendations
1. **Authentication**: Add JWT token authentication
2. **HTTPS**: Use TLS certificates
3. **API Gateway**: Use AWS API Gateway for additional security
4. **Secrets Management**: Use AWS Secrets Manager for sensitive data
5. **Network Security**: Deploy in private subnets with proper security groups

### Adding JWT Authentication (Example)
```python
# Add to requirements.txt
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4

# Add to main.py
from jose import JWTError, jwt
from passlib.context import CryptContext

# Implement token verification middleware
```

## Monitoring and Logging

- **Application Logs**: Stored in `predictions.log`
- **Metrics Endpoint**: `/metrics` provides prediction counts
- **Health Check**: `/health` for container orchestration
- **Structured Logging**: JSON format for log aggregation

## CI/CD Pipeline

The GitHub Actions pipeline:
1. **Test**: Run unit tests and generate coverage
2. **Build**: Create Docker image
3. **Deploy**: Push to AWS ECR (on main branch)

### Required Secrets
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

## Development

### Adding New Features
1. Update the model in `scripts/export_model.py`
2. Modify API endpoints in `app/backend/main.py`
3. Update frontend in `app/frontend/streamlit_app.py`
4. Add tests in `tests/`

### Model Updates
1. Retrain model with new data
2. Update `export_model.py` script
3. Test with existing API
4. Deploy new version

## Troubleshooting

### Common Issues

**Model not found:**
```bash
# Ensure model is exported
python scripts/export_model.py
ls -la model/
```

**API connection error:**
```bash
# Check if backend is running
curl http://localhost:8000/health
```

**Docker build fails:**
```bash
# Clean Docker cache
docker system prune -a
```

## License

This project is for educational and research purposes only. Not intended for clinical use.

## Disclaimer

⚠️ **This application is for research and demonstration purposes only. It should not be used for clinical decision-making or patient care. Always consult qualified healthcare professionals for medical decisions.**