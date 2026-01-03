import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Student Result Predictor", page_icon="🎓")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # Loading the model saved in your notebook: joblib.dump(model, "linear_model.pkl")
    return joblib.load("linear_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error("Model file 'linear_model.pkl' not found. Please run your notebook first.")
    st.stop()

# --- UI HEADER ---
st.title("🎓 Student Pass/Fail Predictor")
st.write("Predict whether a student passes based on study habits and attendance.")
st.divider()

# --- INPUT SECTION ---
st.subheader("Enter Student Details")
col1, col2 = st.columns(2)

with col1:
    hours = st.number_input("Hours Studied", min_value=0.0, max_value=12.0, value=6.0, step=0.5)

with col2:
    attendance = st.number_input("Attendance (%)", min_value=30.0, max_value=100.0, value=75.0, step=1.0)

# --- PREDICTION LOGIC ---
if st.button("Predict Result", type="primary"):
    # Create DataFrame to match the 'feature_names_in_' from your model 
    input_data = pd.DataFrame([[hours, attendance]], columns=['Hours_Studied', 'Attendance'])
    
    # Linear Regression returns a continuous value (e.g., 0.75 or 0.23)
    raw_prediction = model.predict(input_data)[0]
    
    # Thresholding: Since Result is 0 or 1, we use 0.5 as the boundary
    is_pass = raw_prediction >= 0.5

    st.divider()

    # --- DISPLAY RESULTS ---
    if is_pass:
        st.success(f"### Result: PASS ✅")
        st.write(f"The model score is **{raw_prediction:.2f}** (Threshold: 0.5)")
        st.balloons()
    else:
        st.error(f"### Result: FAIL ❌")
        st.write(f"The model score is **{raw_prediction:.2f}** (Threshold: 0.5)")

# --- SIDEBAR ---
st.sidebar.header("Model Info")
st.sidebar.info(
    """
    - **Algorithm:** Linear Regression
    - **Features:** Hours Studied, Attendance
    - **Dataset Size:** 5,000 students
    """
)