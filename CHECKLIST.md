# Manual Setup Checklist

Follow these steps to set up and run the HCT Prediction App locally:

## Prerequisites
- [ ] Python 3.9+ installed
- [ ] Docker and Docker Compose installed (optional but recommended)
- [ ] Git installed

## Setup Steps

### 1. Clone and Navigate
```bash
cd /path/to/project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Export Model (REQUIRED)
```bash
python scripts/export_model.py
```
**Expected output:**
- `model/model.pkl` created
- `model/model_info.pkl` created
- "Model saved to: model/model.pkl" message

### 4. Create Environment File (Optional)
```bash
cp .env.example .env
# Edit .env if needed
```

### 5. Choose Deployment Method

#### Option A: Docker Compose (Recommended)
```bash
docker-compose up --build
```
**Access:**
- Backend API: http://localhost:8000
- Frontend UI: http://localhost:8501
- API Docs: http://localhost:8000/docs

#### Option B: Manual Run
**Terminal 1 - Backend:**
```bash
cd app/backend
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd app/frontend
streamlit run streamlit_app.py
```

### 6. Verify Installation

#### Test Backend API:
```bash
curl http://localhost:8000/health
```
**Expected response:**
```json
{"status": "healthy", "model_loaded": true, "timestamp": "..."}
```

#### Test Prediction:
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

#### Test Frontend:
- [ ] Open http://localhost:8501
- [ ] Fill out the form
- [ ] Click "Predict Survival Probability"
- [ ] Verify results display

### 7. Run Tests (Optional)
```bash
pytest tests/ -v
```

## Troubleshooting

### Model Not Found Error
```bash
# Re-run model export
python scripts/export_model.py
# Check files exist
ls -la model/
```

### Port Already in Use
```bash
# Kill processes on ports 8000 or 8501
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac:
lsof -ti:8000 | xargs kill -9
```

### Docker Issues
```bash
# Clean Docker
docker-compose down
docker system prune -a
docker-compose up --build
```

### Import Errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

## Production Deployment Checklist

### AWS ECR Setup
- [ ] Create ECR repository
- [ ] Configure AWS credentials
- [ ] Push Docker image

### ECS/Fargate Setup
- [ ] Create task definition
- [ ] Configure service
- [ ] Set up load balancer
- [ ] Configure auto-scaling

### Security
- [ ] Enable HTTPS
- [ ] Configure proper CORS
- [ ] Set up authentication (if needed)
- [ ] Review rate limiting settings

### Monitoring
- [ ] Set up CloudWatch logs
- [ ] Configure health checks
- [ ] Set up alerts

## Success Criteria

✅ **Setup Complete When:**
- [ ] Model files exist in `model/` directory
- [ ] Backend API responds to `/health`
- [ ] Frontend loads without errors
- [ ] Test prediction returns valid response
- [ ] All tests pass (if running tests)

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review logs in `predictions.log`
3. Verify all dependencies are installed
4. Ensure model files were generated correctly