import pickle
import pandas as pd
# Load the model to verify it works
with open('Machine_Learning_Models\\heart_disease_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
# Test the loaded model with a sample input
sample_input_1 = {
"Age": 65,
"Sex": "M",
"ChestPainType": "ASY",  
"RestingBP": 160,
"Cholesterol": 360,
"FastingBS": 1,          
"RestingECG": "ST",      
"MaxHR": 110,            
"ExerciseAngina": "Y",   
"Oldpeak": 2.5,          
"ST_Slope": "Flat"       
}



sample_input_2 = {
    "Age": 65,
    "Sex": "M",
    "ChestPainType": "ASY",
    "RestingBP": 160,
    "Cholesterol": 360,
    "FastingBS": 1,
    "RestingECG": "ST",
    "MaxHR": 110,
    "ExerciseAngina": "Y",
    "Oldpeak": 2.5,
    "ST_Slope": "Flat"
}

sample_input_df = pd.DataFrame([sample_input_2])
prediction = loaded_model.predict(sample_input_df)
print(f"Prediction for the sample input: {prediction[0]}")
# The prediction will be either 0 or 1, indicating the presence or absence of heart disease.
if prediction[0] == 1:
    print("The model predicts that the patient has heart disease.")
else:
    print("The model predicts that the patient does not have heart disease.")