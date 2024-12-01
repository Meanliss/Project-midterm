import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import joblib

cvt_date = lambda date_obj: date_obj.strftime("%b-%d") 

def handle_data(input_data):
    #unpacking
    last_credit_pull_d, last_pymnt_d, term, sub_grade = input_data
    loan_df = pd.read_csv("loan_df.csv")

    # Làm tương tự cho các cột khác trong binary_cols

    # Convert inputs to the same format as loan.csv
    last_credit_pull_d = cvt_date(last_credit_pull_d)
    last_pymnt_d       = cvt_date(last_pymnt_d)
    
    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'last_credit_pull_d': [last_credit_pull_d],
        'last_pymnt_d': [last_pymnt_d],
        'term': [term],
        'sub_grade': [sub_grade]
    })

    #label data
    label_encoder = LabelEncoder()

    filter = 'last_credit_pull_d last_pymnt_d term sub_grade'.split(' ')
    loan_df = loan_df[filter]
    loan_df= pd.concat([loan_df,input_data], ignore_index= True) 

    for col in input_data:
        loan_df[col] = label_encoder.fit_transform(loan_df[col])
        input_data[col] = label_encoder.transform(input_data[col])

    for col in input_data:
        freq_encoding = loan_df[col].value_counts().to_dict()
        input_data[col] = input_data[col].map(freq_encoding)


    return input_data

# Kiểm tra và thêm nhãn mới nếu cần



