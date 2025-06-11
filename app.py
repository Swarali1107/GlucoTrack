import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import random
from PIL import Image

# Set layout
st.set_page_config(page_title="GlucoTrack: Diabetes Risk Companion", layout="wide")

# Custom CSS for premium UI
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #f8fafc;
    }
    .stApp {
        background: linear-gradient(135deg, #f9fafb, #f0f4f8);
    }
    .main .block-container {
        padding-top: 3rem;
    }
    .st-bx, .st-c6 {
        background-color: #ffffff !important;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border: 1px solid #e5e7eb;
    }
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        font-weight: 500;
        font-size: 1rem;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(59, 130, 246, 0.25);
    }
    h1, h2, h3, h4 {
        color: #111827;
        font-weight: 600;
    }
    .risk-high {
        color: #ef4444;
        font-weight: 600;
    }
    .risk-low {
        color: #10b981;
        font-weight: 600;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border-left: 4px solid #3b82f6;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load("diabetes_model.pkl")

# Title with hero section
col1, col2 = st.columns([1.5, 1])
with col1:
    st.title("GlucoTrack Diabetes Risk Assessment")
    st.markdown("""
    <p style="font-size: 1.1rem; color: #4b5563;">
        Advanced AI-powered diabetes risk prediction with personalized health insights and 
        actionable recommendations for preventive care.
    </p>
    """, unsafe_allow_html=True)
    
with col2:
    st.image("https://images.unsplash.com/photo-1576091160550-2173dba999ef?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=80", 
             use_container_width=True, caption="Understanding your risk is the first step to prevention")

st.divider()

# Layout
col1, col2 = st.columns([2, 3])

# Inputs
with col1:
    with st.container():
        st.header("Health Profile")
        st.markdown("Complete your health profile for a personalized assessment")
        
        with st.form("form_inputs"):
            preg = st.number_input("Pregnancies", 0, 20, 0, help="Number of times pregnant")
            glucose = st.number_input("Glucose Level (mg/dL)", 0, 300, 90, help="Plasma glucose concentration")
            bp = st.number_input("Blood Pressure (mm Hg)", 0, 200, 70, help="Diastolic blood pressure")
            skin = st.number_input("Skin Thickness (mm)", 0, 100, 20, help="Triceps skin fold thickness")
            insulin = st.number_input("Insulin (mu U/ml)", 0, 1000, 80, help="2-Hour serum insulin")
            bmi = st.number_input("BMI", 0.0, 70.0, 22.0, step=0.1, help="Body mass index")
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.4, step=0.01, 
                                help="Diabetes likelihood based on family history")
            age = st.number_input("Age", 0, 120, 30, help="Age in years")
            
            submitted = st.form_submit_button("Analyze My Risk", 
                                             help="Generate personalized diabetes risk assessment")

input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
input_df = pd.DataFrame(input_data, columns=[
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age"
])

# Prediction
if submitted:
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] * 100

    with col2:
        st.header("Your Diabetes Risk Report")
        
        # Risk summary card
        if pred == 1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin-top: 0;">High Risk Detected</h3>
                <p style="font-size: 2rem; margin-bottom: 0;" class="risk-high">{prob:.1f}% probability</p>
                <p>Based on your health profile, you have an elevated risk for diabetes.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin-top: 0;">Low Risk Detected</h3>
                <p style="font-size: 2rem; margin-bottom: 0;" class="risk-low">{prob:.1f}% probability</p>
                <p>Your current health profile suggests low diabetes risk.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk meter visualization
        st.subheader("Risk Assessment")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            title={'text': "Diabetes Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#3b82f6"},
                'steps': [
                    {'range': [0, 30], 'color': "#d1fae5"},
                    {'range': [30, 70], 'color': "#fef3c7"},
                    {'range': [70, 100], 'color': "#fee2e2"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(t=0, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Health metrics radar
        st.subheader("Health Metrics Overview")
        col_radar1, col_radar2 = st.columns([3, 1])
        
        with col_radar1:
            labels = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 'DPF', 'Age']
            norm_vals = [3, 100, 70, 20, 100, 22, 0.4, 30]
            user_vals = input_data.flatten().tolist()

            df_radar = pd.DataFrame(dict(
                Metric=labels,
                Healthy=norm_vals,
                You=user_vals
            ))

            fig_radar = px.line_polar(df_radar.melt(id_vars="Metric", var_name="Profile", value_name="Value"),
                                    r='Value', theta='Metric', color='Profile', line_close=True,
                                    color_discrete_map={"Healthy": "#10b981", "You": "#3b82f6"})
            fig_radar.update_traces(fill='toself', opacity=0.5)
            fig_radar.update_layout(height=400, legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.1,
                xanchor="center",
                x=0.5
            ))
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with col_radar2:
            st.markdown("""
            <div style="padding: 1rem;">
                <h4 style="margin-top: 0;">Key Metrics</h4>
                <p><strong>Glucose:</strong> Optimal < 100 mg/dL</p>
                <p><strong>Blood Pressure:</strong> Normal < 120/80 mmHg</p>
                <p><strong>BMI:</strong> Healthy range 18.5-24.9</p>
                <p><strong>Insulin:</strong> Fasting < 25 μU/mL</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Health recommendations
        st.subheader("Personalized Recommendations")
        
        if pred == 1:
            rec_col1, rec_col2 = st.columns(2)
            with rec_col1:
                st.markdown("""
                <div style="background-color: #fef2f2; padding: 1.5rem; border-radius: 10px;">
                    <h4 style="margin-top: 0; color: #b91c1c;">Priority Actions</h4>
                    <ul style="padding-left: 1.2rem;">
                        <li>Schedule a consultation with your healthcare provider</li>
                        <li>Begin monitoring fasting glucose levels weekly</li>
                        <li>Reduce refined sugar and processed carbohydrates</li>
                        <li>Incorporate 150 minutes of moderate exercise weekly</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            with rec_col2:
                st.image("https://images.unsplash.com/photo-1498837167922-ddd27525d352?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=80",
                       use_container_width=True, caption="Nutrition plays a key role in diabetes prevention")
            
            st.markdown("""
            <div style="margin-top: 1rem;">
                <h4>Long-term Strategies</h4>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
                    <div style="background-color: #f8fafc; padding: 1rem; border-radius: 8px;">
                        <h5 style="margin: 0 0 0.5rem 0;">Dietary Changes</h5>
                        <p>Focus on whole foods, fiber-rich vegetables, and lean proteins. The Mediterranean diet has shown particular benefits for glucose control.</p>
                    </div>
                    <div style="background-color: #f8fafc; padding: 1rem; border-radius: 8px;">
                        <h5 style="margin: 0 0 0.5rem 0;">Physical Activity</h5>
                        <p>Aim for a combination of aerobic exercise and resistance training. Even short walks after meals can improve glucose metabolism.</p>
                    </div>
                    <div style="background-color: #f8fafc; padding: 1rem; border-radius: 8px;">
                        <h5 style="margin: 0 0 0.5rem 0;">Stress Management</h5>
                        <p>Chronic stress elevates cortisol which can impact glucose levels. Incorporate mindfulness or breathing exercises.</p>
                    </div>
                    <div style="background-color: #f8fafc; padding: 1rem; border-radius: 8px;">
                        <h5 style="margin: 0 0 0.5rem 0;">Sleep Quality</h5>
                        <p>Poor sleep is linked to insulin resistance. Aim for 7-9 hours of quality sleep per night.</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color: #f0fdf4; padding: 1.5rem; border-radius: 10px;">
                <h4 style="margin-top: 0; color: #047857;">Maintenance Plan</h4>
                <p>Your current health profile suggests good metabolic health. Continue these practices:</p>
                <ul style="padding-left: 1.2rem;">
                    <li>Annual preventive health screenings</li>
                    <li>Balanced diet with variety of whole foods</li>
                    <li>Regular physical activity (150+ minutes/week)</li>
                    <li>Maintain healthy weight range</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            st.image("https://images.unsplash.com/photo-1535914254981-b5012eebbd15?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=80",
                   use_container_width=True, caption="Consistency is key for long-term health maintenance")
        
        # BMI analysis
        st.subheader("Body Composition Analysis")
        bmi_col1, bmi_col2 = st.columns([1, 2])
        
        with bmi_col1:
            bmi_value = bmi
            bmi_category = ""
            if bmi_value < 18.5:
                bmi_category = "Underweight"
                color = "#93c5fd"
            elif 18.5 <= bmi_value < 24.9:
                bmi_category = "Normal"
                color = "#86efac"
            elif 25 <= bmi_value < 29.9:
                bmi_category = "Overweight"
                color = "#fcd34d"
            else:
                bmi_category = "Obese"
                color = "#fca5a5"
            
            fig_bmi = go.Figure(go.Indicator(
                mode="number+gauge",
                value=bmi_value,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "BMI"},
                gauge={
                    "shape": "bullet",
                    "axis": {"range": [0, 50]},
                    "threshold": {
                        "line": {"color": "black", "width": 2},
                        "thickness": 0.75,
                        "value": 24.9
                    },
                    "steps": [
                        {"range": [0, 18.5], "color": "#93c5fd"},
                        {"range": [18.5, 24.9], "color": "#86efac"},
                        {"range": [24.9, 29.9], "color": "#fcd34d"},
                        {"range": [29.9, 50], "color": "#fca5a5"}
                    ],
                    "bar": {"color": color}
                }
            ))
            fig_bmi.update_layout(height=150, margin=dict(t=30, b=10))
            st.plotly_chart(fig_bmi, use_container_width=True)
        
        with bmi_col2:
            st.markdown(f"""
            <div style="padding: 1rem;">
                <h4 style="margin-top: 0;">Your BMI: {bmi_value:.1f} ({bmi_category})</h4>
                <p>Body Mass Index (BMI) is a screening tool for weight categories that may lead to health problems.</p>
                {f"<p>Your BMI suggests <strong>{bmi_category.lower()}</strong> status. " + 
                ("Maintaining a healthy weight reduces diabetes risk." if bmi_category == "Normal" else 
                "Consider discussing weight management strategies with your healthcare provider.")}</p>
                <p>Note: BMI doesn't account for muscle mass or fat distribution. Waist circumference may provide additional insight.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Lifestyle score
        st.subheader("Lifestyle Assessment")
        score = 100
        if glucose > 140: score -= 15
        if insulin > 200: score -= 10
        if bmi > 30: score -= 15
        if bp > 130: score -= 10
        if dpf > 0.8: score -= 10
        if age > 50: score -= 10
        
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #3b82f6 {score}%, #e5e7eb {score}%); 
                    height: 24px; border-radius: 12px; margin-bottom: 1rem;"></div>
        <p style="text-align: center; font-size: 1.2rem; font-weight: 500;">
            Lifestyle Score: <span style="color: {'#10b981' if score > 70 else '#ef4444'}">{score}/100</span>
        </p>
        <p style="text-align: center; color: #4b5563;">
            {f"Your lifestyle factors are {'excellent' if score > 85 else 'good' if score > 70 else 'moderate' if score > 50 else 'needing improvement'} based on metabolic health markers"}
        </p>
        """, unsafe_allow_html=True)
        
        # Additional resources
        st.subheader("Additional Resources")
        st.markdown("""
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;">
            <div style="background-color: white; border-radius: 8px; padding: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <h5 style="margin: 0 0 0.5rem 0;">Nutrition Guide</h5>
                <p style="color: #4b5563; margin-bottom: 0.5rem;">Learn about diabetes-friendly foods</p>
                <a href="#" style="color: #3b82f6; text-decoration: none; font-weight: 500;">View Guide →</a>
            </div>
            <div style="background-color: white; border-radius: 8px; padding: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <h5 style="margin: 0 0 0.5rem 0;">Exercise Plans</h5>
                <p style="color: #4b5563; margin-bottom: 0.5rem;">Workouts for metabolic health</p>
                <a href="#" style="color: #3b82f6; text-decoration: none; font-weight: 500;">Explore Options →</a>
            </div>
            <div style="background-color: white; border-radius: 8px; padding: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <h5 style="margin: 0 0 0.5rem 0;">Expert Consult</h5>
                <p style="color: #4b5563; margin-bottom: 0.5rem;">Connect with diabetes specialists</p>
                <a href="#" style="color: #3b82f6; text-decoration: none; font-weight: 500;">Schedule Now →</a>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Empty state
elif not submitted:
    with col2:
        st.subheader("Your Diabetes Risk Report")
        st.markdown("""
        <div style="background-color: #f8fafc; border-radius: 12px; padding: 3rem; text-align: center;">
            <img src="https://images.unsplash.com/photo-1579684385127-1ef15d508118?ixlib=rb-1.2.1&auto=format&fit=crop&w=200&q=80" 
                 style="width: 120px; height: 120px; object-fit: cover; border-radius: 50%; margin-bottom: 1rem;">
            <h4 style="margin: 0.5rem 0; color: #4b5563;">Complete your health profile</h4>
            <p style="color: #6b7280; margin-bottom: 0;">Fill in your health details on the left and click "Analyze My Risk" to generate your personalized diabetes risk assessment and health recommendations.</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #6b7280; font-size: 0.9rem; padding: 1rem;">
    <p>GlucoTrack is designed for informational purposes only and not intended as medical advice. 
    Consult your healthcare provider for personalized medical guidance.</p>
    <p>© 2023 GlucoTrack Health Analytics. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)