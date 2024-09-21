import streamlit as st
import numpy as np
import joblib

# Load the model
model = joblib.load("C:/Users/nirma/Downloads/archive/cancer_severity_rf_model.pkl")

# Define a function to make predictions
def predict_severity(input_data):
    prediction = model.predict([input_data])
    severity_levels = {0: 'Low', 1: 'Medium', 2: 'High'}
    return severity_levels[prediction[0]]

# Streamlit App
st.title("Cancer Severity Prediction")

st.write("Enter the patient's data to predict the severity level.")

# User inputs for each feature
age = st.slider("Age", 1, 100, 30)
gender = st.selectbox("Gender", [0, 1])  # 0: Female, 1: Male
air_pollution = st.slider("Air Pollution", 1, 10, 5)
alcohol_use = st.slider("Alcohol Use", 1, 10, 5)
dust_allergy = st.slider("Dust Allergy", 1, 10, 5)
occupational_hazards = st.slider("Occupational Hazards", 1, 10, 5)
genetic_risk = st.slider("Genetic Risk", 1, 10, 5)
chronic_lung_disease = st.slider("Chronic Lung Disease", 1, 10, 5)
balanced_diet = st.slider("Balanced Diet", 1, 10, 5)
obesity = st.slider("Obesity", 1, 10, 5)
smoking = st.slider("Smoking", 1, 10, 5)
passive_smoker = st.slider("Passive Smoker", 1, 10, 5)
chest_pain = st.slider("Chest Pain", 1, 10, 5)
coughing_of_blood = st.slider("Coughing of Blood", 1, 10, 5)
fatigue = st.slider("Fatigue", 1, 10, 5)
weight_loss = st.slider("Weight Loss", 1, 10, 5)
shortness_of_breath = st.slider("Shortness of Breath", 1, 10, 5)
wheezing = st.slider("Wheezing", 1, 10, 5)
swallowing_difficulty = st.slider("Swallowing Difficulty", 1, 10, 5)
clubbing_of_finger_nails = st.slider("Clubbing of Finger Nails", 1, 10, 5)
frequent_cold = st.slider("Frequent Cold", 1, 10, 5)
dry_cough = st.slider("Dry Cough", 1, 10, 5)
snoring = st.slider("Snoring", 1, 10, 5)

# Input data to pass to the model
input_data = np.array([age, gender, air_pollution, alcohol_use, dust_allergy, 
                       occupational_hazards, genetic_risk, chronic_lung_disease, 
                       balanced_diet, obesity, smoking, passive_smoker, chest_pain, 
                       coughing_of_blood, fatigue, weight_loss, shortness_of_breath, 
                       wheezing, swallowing_difficulty, clubbing_of_finger_nails, 
                       frequent_cold, dry_cough, snoring])

# Prediction button
if st.button("Predict Severity"):
    severity = predict_severity(input_data)
    st.success(f"The predicted severity level is: {severity}")

