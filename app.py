import streamlit as st
import pandas as pd
import joblib

# Load the trained model
# Make sure 'best_model.pickle' is in the same directory as this app.py file
model = joblib.load('best_model.pickle')

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
