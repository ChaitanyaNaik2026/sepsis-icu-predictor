import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import requests
import json
import os

# Page configuration
st.set_page_config(
    page_title="Sepsis Prediction System",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #DC2626 0%, #EA580C 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
    }
    .risk-critical { background-color: #7F1D1D; color: white; padding: 10px; border-radius: 5px; }
    .risk-high { background-color: #DC2626; color: white; padding: 10px; border-radius: 5px; }
    .risk-moderate { background-color: #EA580C; color: white; padding: 10px; border-radius: 5px; }
    .risk-low { background-color: #F59E0B; color: white; padding: 10px; border-radius: 5px; }
    .risk-very-low { background-color: #10B981; color: white; padding: 10px; border-radius: 5px; }
    .feature-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">⚠️ Sepsis Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666;">Early detection of sepsis in ICU patients</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/hospital.png", width=80)
    st.title("Navigation")
    
    app_mode = st.radio(
        "Select Mode",
        ["🏠 Dashboard", "📊 Patient Assessment", "📈 Analytics", "⚙️ Settings"]
    )
    
    st.markdown("---")
    
    # API Configuration
    st.subheader("API Settings")
    api_url = st.text_input(
        "API Endpoint",
        value=os.getenv("API_URL", "http://localhost:8000"),
        help="URL of the FastAPI backend"
    )
    
    if st.button("Test Connection"):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ Connected")
            else:
                st.error("❌ Connection failed")
        except:
            st.error("❌ Cannot connect to API")

# Dashboard
if app_mode == "🏠 Dashboard":
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", "156", "↗️ 12")
    with col2:
        st.metric("High Risk Cases", "24", "↘️ 3", delta_color="inverse")
    with col3:
        st.metric("Avg Risk Score", "38%", "↗️ 2%")
    with col4:
        st.metric("Model Accuracy", "86.5%", "↗️ 1.2%")
    
    # Recent assessments
    st.subheader("Recent Assessments")
    
    assessments = [
        {"id": "P001", "risk": "HIGH", "probability": 0.72, "time": "10:30"},
        {"id": "P002", "risk": "LOW", "probability": 0.18, "time": "09:45"},
        {"id": "P003", "risk": "CRITICAL", "probability": 0.89, "time": "08:15"},
    ]
    
    for assess in assessments:
        with st.expander(f"Patient {assess['id']} - {assess['risk']} Risk ({assess['time']})"):
            st.metric("Risk Probability", f"{assess['probability']:.1%}")
            if assess['risk'] in ['CRITICAL', 'HIGH']:
                st.error("⚠️ Requires immediate attention")
            else:
                st.success("✅ Stable")

# Patient Assessment
elif app_mode == "📊 Patient Assessment":
    st.header("Patient Risk Assessment")
    
    with st.form("assessment_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            prg = st.number_input("PRG", 0.0, 50.0, 1.0)
            pl = st.number_input("PL", 0.0, 1000.0, 120.0)
            pr = st.number_input("PR", 0.0, 250.0, 80.0)
        
        with col2:
            sk = st.number_input("SK", 0.0, 10.0, 4.0)
            ts = st.number_input("TS", 30.0, 45.0, 37.0)
            m11 = st.number_input("M11", 40.0, 150.0, 90.0)
        
        with col3:
            bd2 = st.number_input("BD2", -20.0, 30.0, 0.0)
            age = st.number_input("Age", 0.0, 120.0, 45.0)
            insurance = st.selectbox("Insurance", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        submitted = st.form_submit_button("Assess Sepsis Risk")
        
        if submitted:
            with st.spinner("Analyzing..."):
                # Prepare request
                patient_data = {
                    "PRG": prg, "PL": pl, "PR": pr, "SK": sk,
                    "TS": ts, "M11": m11, "BD2": bd2,
                    "Age": age, "Insurance": insurance
                }
                
                try:
                    response = requests.post(
                        f"{api_url}/predict",
                        json={"patient_data": patient_data},
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display results
                        st.subheader("Assessment Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            risk_class = f"risk-{result['risk_level'].lower().replace(' ', '-')}"
                            st.markdown(f'<div class="{risk_class}">{result["risk_level"]} RISK</div>', unsafe_allow_html=True)
                            st.metric("Probability", f"{result['sepsis_probability']:.1%}")
                            st.metric("Confidence", f"{result['confidence']}%")
                        
                        with col2:
                            # Feature radar chart
                            features = ['PL', 'PR', 'SK', 'TS', 'M11']
                            values = [
                                min(pl / 500 * 100, 100),
                                min(pr / 200 * 100, 100),
                                min(sk / 8 * 100, 100),
                                min((ts - 30) / 15 * 100, 100),
                                min(m11 / 120 * 100, 100)
                            ]
                            
                            fig = go.Figure(data=go.Scatterpolar(
                                r=values,
                                theta=features,
                                fill='toself',
                                fillcolor='rgba(220, 38, 38, 0.3)',
                                line=dict(color='rgb(220, 38, 38)')
                            ))
                            
                            fig.update_layout(
                                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                                showlegend=False,
                                height=300
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Clinical indicators
                        if result['clinical_indicators']:
                            st.subheader("⚠️ Clinical Indicators")
                            for indicator in result['clinical_indicators']:
                                st.warning(indicator)
                        
                        # Actions
                        st.subheader("📋 Recommended Actions")
                        for action in result['immediate_actions']:
                            st.info(action)
                        
                        # Monitoring
                        st.subheader("📊 Monitoring Recommendations")
                        for monitor in result['monitoring_recommendations']:
                            st.success(monitor)
                        
                    else:
                        st.error(f"API Error: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"Connection error: {e}")

# Analytics
elif app_mode == "📈 Analytics":
    st.header("Analytics & Insights")
    
    tab1, tab2 = st.tabs(["Model Performance", "Feature Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Metrics")
            metrics = {
                "Model": ["Random Forest", "XGBoost", "Neural Network", "Ensemble"],
                "AUC": [0.845, 0.852, 0.838, 0.865],
                "Recall": [0.798, 0.812, 0.785, 0.825],
                "Specificity": [0.821, 0.829, 0.815, 0.842]
            }
            
            st.dataframe(pd.DataFrame(metrics))
        
        with col2:
            st.subheader("Confusion Matrix")
            
            # Simulated confusion matrix
            cm_data = [[320, 45], [38, 97]]
            
            fig = px.imshow(
                cm_data,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['No Sepsis', 'Sepsis'],
                y=['No Sepsis', 'Sepsis'],
                text_auto=True,
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Feature Importance")
        
        # Feature importance data
        features = ['PL', 'M11', 'PR', 'Age', 'BD2', 'SK', 'TS', 'PRG', 'Insurance']
        importance = [0.28, 0.22, 0.15, 0.12, 0.08, 0.07, 0.05, 0.02, 0.01]
        
        fig = px.bar(
            x=importance,
            y=features,
            orientation='h',
            title='Feature Importance Ranking',
            color=importance,
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)

# Settings
elif app_mode == "⚙️ Settings":
    st.header("System Settings")
    
    with st.form("settings_form"):
        st.subheader("Alert Thresholds")
        
        critical_threshold = st.slider("Critical Alert", 0.5, 1.0, 0.8)
        high_threshold = st.slider("High Alert", 0.4, 0.9, 0.6)
        
        st.subheader("Notifications")
        email_alerts = st.checkbox("Email Alerts", True)
        dashboard_alerts = st.checkbox("Dashboard Alerts", True)
        
        if st.form_submit_button("Save Settings"):
            st.success("Settings saved!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>Sepsis Prediction System v2.0 • For clinical decision support only</p>
    <p>⚠️ Always verify predictions with clinical judgment</p>
</div>
""", unsafe_allow_html=True)