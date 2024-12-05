import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import joblib
import re

cvt_date = lambda date_obj: date_obj.strftime("%b-%d") 

import shap
def is_valid_zip_code(zip_code): 
    """Checks if the provided zip code is valid.""" 
    return bool(re.match(r'^\d{3}$', zip_code))

def explain_xgb_prediction(model, input_data, top_n=3):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    shap.initjs()
    shap.force_plot(explainer.expected_value, shap_values, input_data)
    
    # Lấy top N đặc trưng quan trọng nhất
    importance_df = pd.DataFrame(list(zip(input_data.columns, shap_values[0])), columns=['feature', 'shap_value'])
    importance_df['abs_shap_value'] = importance_df['shap_value'].abs()
    top_features = importance_df.sort_values(by='abs_shap_value', ascending=False).head(top_n)
    return top_features


def handle_data(new_data):
    print("flag")
    # Create a DataFrame for the input
    input_data = pd.DataFrame([new_data])

    #sync
    input_data.select_dtypes(include=['datetime64[ns]']).apply(cvt_date)
    input_data['emp_length'] = input_data['emp_length'].str.strip().str[0].astype(int)

    
    label_encoder = joblib.load("Transform/label_encoder.joblib")
    freq_encoding = joblib.load("Transform/freq_encoding.joblib")

    binary_cols = ['term', 'sub_grade', 'home_ownership', 'verification_status', 'purpose', 'title', 'zip_code', 'earliest_cr_line', 'last_pymnt_d', 'last_credit_pull_d', 'emp_title'] 
    for col in binary_cols:
        # Nếu giá trị mới không có trong classes_ của LabelEncoder, hãy gán nó là -1 (hoặc giá trị khác tùy chỉnh)
        if input_data[col][0] not in label_encoder.classes_:
            input_data[col] = -1 
        else:
            input_data[col] = label_encoder.transform(input_data[col])

    for col in binary_cols:
        # Nếu giá trị mới không có trong frequency_encoding, hãy gán nó là 0 (hoặc giá trị khác tùy chỉnh)
        if input_data[col][0] not in freq_encoding:
            input_data[col] = 0 
        else:
            input_data[col] = freq_encoding[input_data[col][0]]

    for key in input_data:
        print(f" {key}  : {input_data[key]}")
    return input_data

# Kiểm tra và thêm nhãn mới nếu cần




