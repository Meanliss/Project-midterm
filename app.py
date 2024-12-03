import joblib 
import pandas as pd
import streamlit as st
import numpy as np
import datetime
import sklearn
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt
import os
from PrePrcs import handle_data

    
#load model
dt_path = "Model\dt_model.pkl"
# lr_path = "Model\logistic_model.pkl"
# nn_path =  "Model\neural_network_model.pkl"
dt_model = joblib.load(dt_path)

today = datetime.date.today()



#=============================================welcome titling=====================================================
st.title("loan acceptance prediction")
st.markdown(" predict loan application acceptance or rejection based on customer details")
st.header('Customer Details')

#=============================================Input fields========================================================
last_credit_pull_d = st.date_input('Last Credit Pull Date', 
                                   datetime.date(2024, 1, 1),
                                   max_value=today)
last_pymnt_d = st.date_input('Last Payment Date', 
                             datetime.date(2024, 1, 1),
                             max_value=today)
term = st.selectbox('Term', 
                    ['36 months', '60 months'])
sub_grade = st.selectbox('Sub Grade', 
                         [f'{chr(i)}{j}' for i in range(ord('A'), ord('H')) for j in range(1, 6)])

#convert data into a type that fit the model
input_data = handle_data([last_credit_pull_d,
                          last_pymnt_d,
                          term,
                          sub_grade])

#=============================================logging============================================================
print(f"{last_credit_pull_d} : {type(last_credit_pull_d)}")
print(f"{last_pymnt_d}: {type(last_pymnt_d)}")
print(f"{term}: {type(term)}")
print(f"{sub_grade} : {type(sub_grade)}")
filter = 'last_credit_pull_d last_pymnt_d term sub_grade'.split(' ')
print(input_data[filter])

#==================================Predict and display the result=================================================
if st.button('Predict'):
    print(f" predict in : {input_data.values.reshape(1,-1)}")
    prediction = dt_model.predict(input_data.values.reshape(1,-1))
    print(prediction)
    st.success(f'The loan approval prediction is: {"Approved" if prediction[0] == 1 else "Not Approved"}')

    # # Giải thích dự đoán với SHAP
    # explainer = joblib.load("explainer_dt.pkl")  # Load explainer đã huấn luyện
    # shap_values = explainer(input_data)

    # # Biểu đồ Waterfall
    # st.subheader("Explanation of the Prediction")
    # st.markdown("Below is the contribution of each feature to the model's decision:")

    # # Waterfall plot
    # fig, ax = plt.subplots(figsize=(10, 6))
    # shap.waterfall_plot(shap_values[0], show=False)  # Chọn giá trị đầu tiên (dữ liệu input của người dùng)
    # st.pyplot(fig)

