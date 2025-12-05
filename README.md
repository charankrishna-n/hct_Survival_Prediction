HOW TO RUN THE FILE

Step 1: Install dependencies
pip install -r requirements.txt

Step 2: Export the model
python scripts/export_model.py

Step 3: Run backend (Terminal 1)
cd app/backend
python main.py

Step 4: Run frontend (Terminal 2)
cd app/frontend
streamlit run streamlit_app.py

Alternatively, if you have Docker Desktop installed, use:
docker compose up --build
