import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Heart Disease Risk Assessment",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the title
st.markdown("""
    <style>
    .title {
        font-size: 2.5em;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 0.5em;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .subtitle {
        font-size: 1.2em;
        color: #666;
        text-align: center;
        margin-bottom: 2em;
    }
    .heart-icon {
        font-size: 1.5em;
        vertical-align: middle;
        margin: 0 10px;
    }
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    .stNumberInput>div>div>input {
        border-radius: 5px;
    }
    .stSelectbox>div>div>div {
        border-radius: 5px;
    }
    .risk-high {
        color: #ff4b4b;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffa500;
        font-weight: bold;
    }
    .risk-low {
        color: #00cc00;
        font-weight: bold;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted black;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model and scaler with caching
@st.cache_resource
def load_model():
    try:
        model_path = Path(__file__).parent / 'models' / 'best_heart_disease_model.pkl'
        scaler_path = Path(__file__).parent / 'models' / 'scaler.pkl'
        
        if not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError("Model files not found")
            
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error("Error loading the prediction model. Please try again later.")
        st.stop()

# Initialize session state
if 'form_data' not in st.session_state:
    st.session_state.form_data = {}

# Title with heart icons
st.markdown("""
    <div class="title">
        <span class="heart-icon">❤️</span>
        Heart Disease Risk Assessment
        <span class="heart-icon">❤️</span>
    </div>
    <div class="subtitle">
        Enter your health information below to assess your risk of heart disease
    </div>
""", unsafe_allow_html=True)

# Add a horizontal line for visual separation
st.markdown("---")

# Create form
with st.form("heart_disease_form"):
    st.header("Personal Information")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=120,
            value=50,
            help="Enter your age in years. Age is an important factor in heart disease risk assessment."
        )
        
    with col2:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
            help="Select your gender. Men generally have a higher risk of heart disease than women."
        )
        gender_encoded = 1 if gender == "Male" else 0
    
    st.header("Medical History")
    col1, col2 = st.columns(2)
    
    with col1:
        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
            help="""Select the type of chest pain you experience:
            - Typical Angina: Chest pain that occurs with activity and is relieved by rest
            - Atypical Angina: Chest pain that doesn't follow the typical pattern
            - Non-Anginal Pain: Chest pain not related to the heart
            - Asymptomatic: No chest pain"""
        )
        chest_pain_encoded = {
            "Typical Angina": 1,
            "Atypical Angina": 2,
            "Non-Anginal Pain": 3,
            "Asymptomatic": 4
        }[chest_pain]
        
    with col2:
        resting_bp = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=90,
            max_value=200,
            value=120,
            help="Enter your resting blood pressure in mm Hg. Normal range is 90-120 mm Hg. High blood pressure increases heart disease risk."
        )
    
    st.header("Blood Test Results")
    col1, col2 = st.columns(2)
    
    with col1:
        cholesterol = st.number_input(
            "Serum Cholesterol (mg/dl)",
            min_value=100,
            max_value=600,
            value=200,
            help="Enter your total cholesterol level in mg/dl. Normal range is 125-200 mg/dl. High cholesterol increases heart disease risk."
        )
        
    with col2:
        fasting_blood_sugar = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            ["No", "Yes"],
            help="Indicate if your fasting blood sugar is above 120 mg/dl. High blood sugar can indicate diabetes, which increases heart disease risk."
        )
        fasting_blood_sugar_encoded = 1 if fasting_blood_sugar == "Yes" else 0
    
    st.header("Electrocardiogram (ECG) Results")
    col1, col2 = st.columns(2)
    
    with col1:
        resting_ecg = st.selectbox(
            "Resting ECG Results",
            ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
            help="""Select your resting ECG results:
            - Normal: No abnormalities detected
            - ST-T Wave Abnormality: May indicate heart muscle strain
            - Left Ventricular Hypertrophy: Thickening of the heart's main pumping chamber"""
        )
        resting_ecg_encoded = {
            "Normal": 0,
            "ST-T Wave Abnormality": 1,
            "Left Ventricular Hypertrophy": 2
        }[resting_ecg]
        
    with col2:
        max_heart_rate = st.number_input(
            "Maximum Heart Rate Achieved",
            min_value=60,
            max_value=220,
            value=150,
            help="Enter your maximum heart rate achieved during exercise. Normal range varies by age. Lower values may indicate heart problems."
        )
    
    st.header("Exercise Test Results")
    col1, col2 = st.columns(2)
    
    with col1:
        exercise_angina = st.selectbox(
            "Exercise Induced Angina",
            ["No", "Yes"],
            help="Indicate if you experience chest pain during exercise. This is a significant risk factor for heart disease."
        )
        exercise_angina_encoded = 1 if exercise_angina == "Yes" else 0
        
    with col2:
        st_depression = st.number_input(
            "ST Depression During Exercise",
            min_value=0.0,
            max_value=6.0,
            value=0.0,
            step=0.1,
            help="Enter the amount of ST segment depression during exercise. Values above 2.0 mm may indicate heart problems."
        )
        st_slope = st.selectbox(
            "Slope of Peak Exercise ST Segment",
            ["Upsloping", "Flat", "Downsloping"],
            help="""Select the slope of your ST segment during peak exercise:
            - Upsloping: Generally normal
            - Flat: May indicate heart problems
            - Downsloping: Often indicates significant heart problems"""
        )
        st_slope_encoded = {
            "Upsloping": 1,
            "Flat": 2,
            "Downsloping": 3
        }[st_slope]
    
    st.header("Additional Tests")
    col1, col2 = st.columns(2)
    
    with col1:
        num_vessels = st.selectbox(
            "Number of Major Vessels Colored by Fluoroscopy",
            ["0", "1", "2", "3"],
            help="""Select the number of major blood vessels visible on fluoroscopy:
            - 0: No significant blockages
            - 1-3: Increasing number of blocked or narrowed arteries"""
        )
        num_vessels_encoded = int(num_vessels)
        
    with col2:
        thal = st.selectbox(
            "Thalassemia",
            ["Normal", "Fixed Defect", "Reversible Defect"],
            help="""Select your thalassemia test results:
            - Normal: No blood flow problems
            - Fixed Defect: Permanent damage to heart muscle
            - Reversible Defect: Temporary blood flow problems"""
        )
        thal_encoded = {
            "Normal": 3,
            "Fixed Defect": 6,
            "Reversible Defect": 7
        }[thal]
    
    # Submit button
    submitted = st.form_submit_button("Predict Risk")
    
    if submitted:
        try:
            # Prepare input data with correct feature order and names
            input_data = pd.DataFrame({
                'age': [age],
                'sex': [gender_encoded],
                'cp': [chest_pain_encoded],
                'trestbps': [resting_bp],
                'chol': [cholesterol],
                'fbs': [fasting_blood_sugar_encoded],
                'restecg': [resting_ecg_encoded],
                'thalach': [max_heart_rate],
                'exang': [exercise_angina_encoded],
                'oldpeak': [st_depression],
                'slope': [st_slope_encoded],
                'ca': [num_vessels_encoded],
                'thal': [thal_encoded],
                'age_group_<40': [1 if age < 40 else 0],
                'age_group_40-55': [1 if 40 <= age <= 55 else 0],
                'age_group_55-65': [1 if 55 < age <= 65 else 0],
                'age_group_>65': [1 if age > 65 else 0],
                'bmi_proxy': [cholesterol / (age * 0.1)],
                'bp_risk': [1 if (resting_bp > 140) or (max_heart_rate < 60) else 0]
            }, index=[0])  # Add index to ensure proper feature names
            
            # Load model and scaler
            model, scaler = load_model()
            
            # Scale the input data
            input_scaled = scaler.transform(input_data)
            input_scaled = pd.DataFrame(input_scaled, columns=input_data.columns)  # Preserve feature names
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]
            
            # Display results
            st.header("Prediction Results")
            
            # Risk level
            if probability >= 0.7:
                risk_level = "High"
                risk_class = "risk-high"
            elif probability >= 0.4:
                risk_level = "Medium"
                risk_class = "risk-medium"
            else:
                risk_level = "Low"
                risk_class = "risk-low"
            
            st.markdown(f"""
                <div style='text-align: center;'>
                    <h2>Your Risk Level: <span class='{risk_class}'>{risk_level}</span></h2>
                    <h3>Probability: {probability:.1%}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            # Risk factors
            st.subheader("Key Risk Factors")
            risk_factors = []
            
            if age > 45:
                risk_factors.append("Age over 45")
            if resting_bp > 140:
                risk_factors.append("High blood pressure")
            if cholesterol > 240:
                risk_factors.append("High cholesterol")
            if max_heart_rate < 100:
                risk_factors.append("Low maximum heart rate")
            if exercise_angina == "Yes":
                risk_factors.append("Exercise-induced angina")
            if st_depression > 2:
                risk_factors.append("Significant ST depression")
            
            if risk_factors:
                st.write("The following factors contribute to your risk:")
                for factor in risk_factors:
                    st.write(f"- {factor}")
            else:
                st.write("No significant risk factors identified.")
            
            # Recommendations
            st.subheader("Recommendations")
            if risk_level == "High":
                st.write("""
                    - Schedule an appointment with your doctor immediately
                    - Consider lifestyle changes (diet, exercise)
                    - Monitor your blood pressure regularly
                    - Avoid smoking and excessive alcohol consumption
                """)
            elif risk_level == "Medium":
                st.write("""
                    - Schedule a check-up with your doctor
                    - Maintain a healthy diet and regular exercise
                    - Monitor your blood pressure and cholesterol
                    - Consider stress management techniques
                """)
            else:
                st.write("""
                    - Continue with regular health check-ups
                    - Maintain a healthy lifestyle
                    - Stay active and exercise regularly
                    - Eat a balanced diet
                """)
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            st.error("An error occurred during prediction. Please try again later.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center;'>
        <p>This tool is for educational purposes only and should not replace professional medical advice.</p>
        <p>Always consult with your healthcare provider for medical concerns.</p>
    </div>
    """, unsafe_allow_html=True) 