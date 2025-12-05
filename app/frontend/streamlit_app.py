"""
Streamlit frontend for HCT survival prediction.
⚠️  FOR RESEARCH/DEMO USE ONLY. NOT FOR CLINICAL DECISION-MAKING.
"""
import os
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="HCT Survival Prediction",
    page_icon="🏥",
    layout="wide"
)

# API endpoint
API_URL = os.getenv("API_URL", "http://localhost:8000")

def main():
    st.title("🏥 HCT Survival Prediction")
    
    # Prominent disclaimer
    st.error("⚠️ **DISCLAIMER: FOR RESEARCH/DEMO USE ONLY. NOT FOR CLINICAL DECISION-MAKING.**")
    
    st.markdown("""
    This tool predicts post-transplant survival probability based on patient characteristics.
    It uses an XGBoost model trained on synthetic data for demonstration purposes.
    """)
    
    # Check API health
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        if health_response.status_code != 200:
            st.error("❌ Backend API is not available")
            return
    except requests.exceptions.RequestException:
        st.error("❌ Cannot connect to backend API. Make sure it's running on localhost:8000")
        return
    
    st.success("✅ Connected to prediction API")
    
    # Input form
    st.header("Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=80, value=45)
        gender = st.selectbox("Gender", ["Male", "Female"])
        donor_type = st.selectbox(
            "Donor Type", 
            ["Matched sibling", "Matched unrelated", "Haploidentical", "Cord blood"]
        )
        comorbidity_score = st.number_input("Comorbidity Score", min_value=0, max_value=10, value=2)
        disease_type = st.selectbox("Disease Type", ["AML", "ALL", "MDS", "Lymphoma", "Other"])
    
    with col2:
        conditioning_intensity = st.selectbox(
            "Conditioning Intensity", 
            ["Myeloablative", "Reduced-intensity"]
        )
        prior_transplants = st.selectbox("Prior Transplants", [0, 1])
        time_from_diagnosis_days = st.number_input(
            "Days from Diagnosis", min_value=1, max_value=5000, value=180
        )
        treatment_days = st.number_input(
            "Treatment Days", min_value=1, max_value=365, value=30
        )
    
    # Predict button
    if st.button("🔮 Predict Survival Probability", type="primary"):
        # Prepare data
        patient_data = {
            "age": age,
            "gender": gender,
            "donor_type": donor_type,
            "comorbidity_score": comorbidity_score,
            "disease_type": disease_type,
            "conditioning_intensity": conditioning_intensity,
            "prior_transplants": prior_transplants,
            "time_from_diagnosis_days": time_from_diagnosis_days,
            "treatment_days": treatment_days
        }
        
        # Make prediction
        try:
            with st.spinner("Making prediction..."):
                response = requests.post(f"{API_URL}/predict", json=patient_data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                
                # Display results
                st.header("Prediction Results")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Probability gauge
                    prob = result["probability"]
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = prob * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Survival Probability (%)"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 100], 'color': "gray"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=300)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    st.metric("Prediction", result["prediction"])
                
                with col2:
                    # Feature importance
                    if "feature_importance" in result["explainability"]:
                        importance_data = result["explainability"]["feature_importance"]
                        if isinstance(importance_data, dict) and len(importance_data) > 0:
                            df_importance = pd.DataFrame(
                                list(importance_data.items()), 
                                columns=["Feature", "Importance"]
                            )
                            
                            fig_bar = px.bar(
                                df_importance, 
                                x="Importance", 
                                y="Feature",
                                orientation='h',
                                title="Top Contributing Features"
                            )
                            fig_bar.update_layout(height=300)
                            st.plotly_chart(fig_bar, use_container_width=True)
                        else:
                            st.info("Feature importance not available")
                    
                # Show raw response
                with st.expander("View Raw Response"):
                    st.json(result)
                
                # Disclaimer again
                st.warning("⚠️ This prediction is for research/demo purposes only and should not be used for clinical decision-making.")
                
            elif response.status_code == 422:
                st.error("❌ Invalid input data. Please check your entries.")
                st.json(response.json())
            else:
                st.error(f"❌ Prediction failed: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Connection error: {e}")
    
    # API metrics
    st.sidebar.header("API Status")
    try:
        metrics_response = requests.get(f"{API_URL}/metrics", timeout=5)
        if metrics_response.status_code == 200:
            metrics = metrics_response.json()
            st.sidebar.metric("Predictions Made", metrics.get("prediction_count", 0))
            st.sidebar.metric("Errors", metrics.get("error_count", 0))
    except:
        st.sidebar.error("Cannot fetch metrics")

if __name__ == "__main__":
    main()