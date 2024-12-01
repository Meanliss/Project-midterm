import joblib 
import pandas as pd
import streamlit as st
import numpy as np
import datetime
import sklearn
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt

#load model
dt_path = f"Model\dt_model.pkl"
lr_path = f"Model\logistic_model.pkl"
nn_path = f"Model\neural_network_model.pkl"
dt_model = joblib.load(dt_path)
lr_model = joblib.load(lr_path)

st.title("loan acceptance prediction")
st.markdown(" predict loan application acceptance or rejection based on customer details")

st.header('Customer Details')

# Input fields
last_credit_pull_d = st.date_input('Last Credit Pull Date', datetime.date(2024, 1, 1))
last_pymnt_d = st.date_input('Last Payment Date', datetime.date(2024, 1, 1))
term = st.selectbox('Term', ['36 months', '60 months'])
sub_grade = st.selectbox('Sub Grade', [f'{chr(i)}{j}' for i in range(ord('A'), ord('H')) for j in range(1, 6)])

# Convert inputs to the required format
term = int(term.split()[0])
last_credit_pull_d = pd.to_datetime(last_credit_pull_d).toordinal()
last_pymnt_d = pd.to_datetime(last_pymnt_d).toordinal()

# Create a DataFrame for the input
input_data = pd.DataFrame({
    'last_credit_pull_d': [last_credit_pull_d],
    'last_pymnt_d': [last_pymnt_d],
    'term': [term],
    'sub_grade': [sub_grade]
})


#transform a bit
# Label Encoding for binary categorical columns
label_encoder = LabelEncoder()
for col in input_data:
    input_data[col] = label_encoder.fit_transform(input_data[col])

for col in input_data:
    freq_encoding = input_data[col].value_counts().to_dict()
    input_data[col] = input_data[col].map(freq_encoding)

print(input_data)

# Predict and display the result
if st.button('Predict'):
    prediction = dt_model.predict(input_data.values.reshape(1,-1))
    print(prediction)
    st.success(f'The loan approval prediction is: {"Approved" if prediction[0] == 1 else "Not Approved"}')



