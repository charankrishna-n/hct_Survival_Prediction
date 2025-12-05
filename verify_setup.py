#!/usr/bin/env python3
"""
Quick verification script to check if the project is set up correctly.
Run this after following the setup instructions.
"""
import os
import sys
import importlib.util

def check_file_exists(filepath, description):
    """Check if a file exists and print status"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} (MISSING)")
        return False

def check_python_package(package_name):
    """Check if a Python package is installed"""
    spec = importlib.util.find_spec(package_name)
    if spec is not None:
        print(f"✅ Python package: {package_name}")
        return True
    else:
        print(f"❌ Python package: {package_name} (NOT INSTALLED)")
        return False

def main():
    print("🔍 HCT Prediction App Setup Verification")
    print("=" * 50)
    
    all_good = True
    
    # Check critical files
    critical_files = [
        ("requirements.txt", "Requirements file"),
        ("scripts/export_model.py", "Model export script"),
        ("app/backend/main.py", "Backend API"),
        ("app/frontend/streamlit_app.py", "Frontend app"),
        ("Dockerfile", "Docker configuration"),
        ("docker-compose.yml", "Docker Compose configuration"),
        ("tests/test_api.py", "API tests"),
        ("README.md", "Documentation")
    ]
    
    print("\n📁 File Structure Check:")
    for filepath, description in critical_files:
        if not check_file_exists(filepath, description):
            all_good = False
    
    # Check Python packages
    required_packages = [
        "fastapi", "uvicorn", "pydantic", "streamlit", 
        "pandas", "numpy", "sklearn", "xgboost", 
        "joblib", "matplotlib", "pytest"
    ]
    
    print("\n📦 Python Dependencies Check:")
    for package in required_packages:
        if not check_python_package(package):
            all_good = False
    
    # Check model files
    print("\n🤖 Model Files Check:")
    model_files = [
        ("model/model.pkl", "Trained model"),
        ("model/model_info.pkl", "Model metadata")
    ]
    
    model_exists = True
    for filepath, description in model_files:
        if not check_file_exists(filepath, description):
            model_exists = False
    
    if not model_exists:
        print("\n⚠️  Model files not found. Run: python scripts/export_model.py")
        all_good = False
    
    # Test model loading if files exist
    if model_exists:
        try:
            import joblib
            model = joblib.load("model/model.pkl")
            model_info = joblib.load("model/model_info.pkl")
            print("✅ Model loading test: SUCCESS")
        except Exception as e:
            print(f"❌ Model loading test: FAILED ({e})")
            all_good = False
    
    # Final status
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 Setup verification PASSED!")
        print("\nNext steps:")
        print("1. Run: python scripts/export_model.py (if model files missing)")
        print("2. Run: docker-compose up --build")
        print("3. Access: http://localhost:8501 (frontend)")
        print("4. Access: http://localhost:8000/docs (API docs)")
    else:
        print("❌ Setup verification FAILED!")
        print("\nPlease fix the issues above and run this script again.")
        print("Refer to README.md and CHECKLIST.md for detailed instructions.")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())