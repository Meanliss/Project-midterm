import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import joblib

cvt_date = lambda date_obj: date_obj.strftime("%b-%d") 

def explain_decision_tree_prediction(last_credit_pull_d, last_pymnt_d, term, sub_grade, prediction, model):
    """
    Explains the Decision Tree model's prediction (0: rejected, 1: accepted)
    based on the input features.
    """

    if prediction == 0:  # Loan Rejected
        explanation = "The loan application was likely rejected due to the following factors:\n"

        if last_credit_pull_d < model.tree_.threshold[0]:
            explanation += "- **last_credit_pull_d:** The recent credit inquiry indicates potential financial instability.\n"
        else:
            explanation += "- **last_credit_pull_d:** The credit inquiry timing is favorable.\n"

        if last_pymnt_d < model.tree_.threshold[1]:
            explanation += "- **last_pymnt_d:** The recent last payment date might raise concerns about repayment capacity.\n"
        else:
            explanation += "- **last_pymnt_d:** The last payment date is not a major concern.\n"

        if term < model.tree_.threshold[2]:
            explanation += "- **term:** The shorter loan term could suggest higher risk.\n"
        else:
            explanation += "- **term:** The longer loan term is considered less risky.\n"

        if sub_grade < model.tree_.threshold[3]:
            explanation += "- **sub_grade:** The lower sub_grade indicates a higher risk profile.\n"
        else:
            explanation += "- **sub_grade:** The sub_grade is acceptable.\n"

        return explanation

    else:  # Loan Accepted
        return "The loan application was likely accepted based on the provided features."

def handle_data(input_data):
    print("flag")
    #unpacking
    last_credit_pull_d, last_pymnt_d, term, sub_grade = input_data
    loan_df = pd.read_csv("Data/loan_df.csv")

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

    #add the data point in the bottoms of the df
    filter = 'last_credit_pull_d last_pymnt_d term sub_grade'.split(' ')
    loan_df = loan_df[filter]
    loan_df= pd.concat([loan_df,input_data], ignore_index= True) 


    for col in input_data:
        loan_df[col] = label_encoder.fit_transform(loan_df[col])
        input_data[col] = label_encoder.transform(input_data[col])

    for col in input_data:
        freq_encoding = loan_df[col].value_counts().to_dict()
        freq_encoding = loan_df[col].map(freq_encoding)
        input_data[col] = input_data[col].map(freq_encoding).fillna(0) 

    mapping_term = {
        '36 months' : 26678,
        '60 months' : 9047,
        'NaN' : 1091
    }

    input_data['term'] = mapping_term[term]
    return input_data

# Kiểm tra và thêm nhãn mới nếu cần




