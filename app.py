import streamlit as st
import pandas as pd
import joblib
import os
import pickle

# Load the trained model
# Make sure 'best_model.pickle' is in the same directory as this app.py file
MODEL_PATH = 'best_model.pickle'

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file '{MODEL_PATH}' not found!")
    st.warning("To fix this issue:")
    st.info("""
    1. Make sure you have the `student_habits_performance.csv` dataset file
    2. Run all cells in the Untitled13.ipynb notebook in order
    3. This will train the model and save it as 'best_model.pickle'
    4. Then refresh this Streamlit app
    """)
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except (EOFError, pickle.UnpicklingError, AttributeError, KeyError, ValueError) as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.warning("The best_model.pickle file is corrupted. To fix:")
    st.info("""
    1. Delete the corrupted best_model.pickle file
    2. Obtain the student_habits_performance.csv dataset
    3. Run all cells in Untitled13.ipynb to retrain and save the model
    4. Refresh this Streamlit app
    """)
    st.stop()

st.title('Student Exam Score Predictor')
st.write('Enter student details to predict their exam score.')

# Input features
study_hours = st.slider('Study Hours per Day', 0.0, 10.0, 3.5, 0.1)
attendance = st.slider('Attendance Percentage', 0.0, 100.0, 85.0, 0.1)
mental_health = st.slider('Mental Health Rating (1-10)', 1, 10, 5)
sleep_hours = st.slider('Sleep Hours per Night', 3.0, 10.0, 7.0, 0.1)
part_time_job_str = st.radio('Part-time Job?', ('No', 'Yes'))

# Preprocessing input (similar to how it was done in the notebook)
# 'part_time_job' was LabelEncoded: No=0, Yes=1
part_time_job_encoded = 1 if part_time_job_str == 'Yes' else 0

# Create a DataFrame from the inputs
input_data = pd.DataFrame([[study_hours, attendance, mental_health, sleep_hours, part_time_job_encoded]],
                           columns=['study_hours_per_day', 'attendance_percentage', 'mental_health_rating', 'sleep_hours', 'part_time_job'])

# Make prediction
if st.button('Predict Exam Score'):
    prediction = model.predict(input_data)[0]
    st.success(f'Predicted Exam Score: {prediction:.2f}')
