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
from PrePrcs import handle_data, explain_xgb_prediction, is_valid_zip_code
import xgboost as xgb
from xgboost import XGBClassifier
#=========================================load explaination for eachs features ===================================
FEATURE_EXPLANATIONS = {
    "last_pymnt_d": "The date of your last payment. If your last payment was a long time ago, this can negatively impact your credit evaluation.",
    "last_credit_pull_d": "The date when your credit information was last checked. If this information is outdated, it may raise concerns about your current financial status.",
    "sub_grade": "This indicates the risk level of the loan. Loans with a lower sub_grade are often seen as riskier and have higher interest rates.",
    "zip_code": "Your postal code. This can be related to the economic status of the area where you live.",
    "int_rate": "The interest rate of the loan. A higher interest rate can increase the overall cost of the loan and impact your ability to repay.",
    "revol_bal": "Your revolving balance. A high revolving balance can indicate that you are using a lot of credit and may have difficulty repaying.",
    "purpose": "The purpose of the loan. Certain loan purposes may be seen as riskier and affect the lending decision.",
    "earliest_cr_line": "The date when you first opened a credit account. A longer credit history is often viewed positively.",
    "revol_util": "The revolving credit utilization rate. A high utilization rate can indicate higher credit risk.",
    "term": "The term of the loan. A longer term can mean more interest paid over time.",
    "dti": "The debt-to-income ratio. A higher ratio can indicate that you have a lot of debt relative to your income, which may lower your ability to repay.",
    "last_pymnt_amnt": "The amount of your last payment. A low last payment amount can negatively affect your credit evaluation.",
    "total_rec_int": "The total received interest. This can indicate your creditworthiness.",
    "emp_title": "Your job title. Certain job titles may be viewed as higher risk by lenders.",
    "title": "The title of your loan. An unclear or concerning title can affect the lending decision.",
    "total_pymnt": "The total payment amount. A low total payment amount can increase lending risk.",
    "loan_amnt": "The amount of the loan. A higher loan amount can increase credit risk.",
    "total_rec_prncp": "The total principal received. This can indicate your creditworthiness.",
    "annual_inc": "Your annual income. A lower income can negatively impact your ability to repay.",
    "total_acc": "The total number of accounts. A larger number of accounts can indicate more credit experience.",
    "funded_amnt": "The funded amount. A higher funded amount can increase lending risk.",
    "home_ownership": "Your home ownership status. Owning a home is often seen as a positive financial signal.",
    "open_acc": "The number of open accounts. A larger number of open accounts can indicate good credit management experience.",
    "verification_status": "The verification status. The verification status of your credit can impact the lending decision.",
    "emp_length": "The length of employment. A longer employment history is often viewed positively regarding financial stability."
}

#============================================load model==========================================================
xgb_path = "Model/xgb_model.json"
xgb_model = XGBClassifier() 
xgb_model.load_model(xgb_path)

today = datetime.date.today()



#=============================================welcome titling=====================================================
st.title("loan acceptance prediction")
st.markdown(" predict loan application acceptance or rejection based on customer details")
st.header('Customer Details')

#=============================================Input fields========================================================

# Chia giao diện thành 2 cột
col1, col2 = st.columns(2)

# Cột 1: Ngày tháng và các thuộc tính phân loại
with col1:
    st.header("Thông tin phân loại")
    last_pymnt_d = st.date_input("Last Payment Date", value=datetime.date.today(), max_value=today)
    last_credit_pull_d = st.date_input("Last Credit Pull Date", value=datetime.date.today(), max_value=today)
    earliest_cr_line = st.date_input("Earliest Credit Line", value=datetime.date.today(), max_value=today)
    sub_grade = st.selectbox('Sub Grade', 
                         [f'{chr(i)}{j}' for i in range(ord('A'), ord('H')) for j in range(1, 6)])
    zip_code = st.text_input("ZIP Code (first 3 digits)")
    purpose = st.selectbox("Purpose", ['debt_consolidation', 'credit_card', 'other', 
                                       'home_improvement','major_purchase', 'small_business', 
                                       'car', 'wedding', 'medical', 
                                       'moving', 'vacation', 'house',
                                        'educational', 'renewable_energy'])
    emp_title = st.text_input("Employment Title")
    title = st.text_input("Loan Title")
    home_ownership = st.selectbox("Home Ownership", ["OWN", "MORTGAGE", "RENT", "OTHER"])
    verification_status = st.selectbox("Verification Status", ["Verified", "Source Verified", "Not Verified"])
    emp_length = st.selectbox("Employment Length", [f"{i} years" for i in range(1, 11)] + ["< 1 year"])

# Cột 2: Các thuộc tính số
with col2:
    st.header("Thông tin số liệu")
    int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, step=0.1)
    revol_bal = st.number_input("Revolving Balance", min_value=0.0, step=100.0)
    revol_util = st.number_input("Revolving Utilization Rate (%)", min_value=0.0, max_value=100.0, step=0.1)
    term = st.selectbox("Loan Term", ["36 months", "60 months"])
    dti = st.number_input("Debt-to-Income Ratio", min_value=0.0, step=0.1)
    last_pymnt_amnt = st.number_input("Last Payment Amount", min_value=0.0, step=100.0)
    total_rec_int = st.number_input("Total Received Interest", min_value=0.0, step=100.0)
    total_pymnt = st.number_input("Total Payment", min_value=0.0, step=100.0)
    loan_amnt = st.number_input("Loan Amount", min_value=0.0, step=100.0)
    total_rec_prncp = st.number_input("Total Received Principal", min_value=0.0, step=100.0)
    annual_inc = st.number_input("Annual Income", min_value=0.0, step=1000.0)
    total_acc = st.number_input("Total Accounts", min_value=0, step=1)
    funded_amnt = st.number_input("Funded Amount", min_value=0.0, step=100.0)
    open_acc = st.number_input("Open Accounts", min_value=0, step=1)

# Hiển thị kết quả đầu ra

#=============================================logging============================================================
result = {
        "last_pymnt_d": last_pymnt_d,
        "last_credit_pull_d": last_credit_pull_d,
        "sub_grade": sub_grade,
        "zip_code": zip_code,
        "int_rate": int_rate,
        "revol_bal": revol_bal,
        "purpose": purpose,
        "earliest_cr_line": earliest_cr_line,
        "revol_util": revol_util,
        "term": term,
        "dti": dti,
        "last_pymnt_amnt": last_pymnt_amnt,
        "total_rec_int": total_rec_int,
        "emp_title": emp_title,
        "title": title,
        "total_pymnt": total_pymnt,
        "loan_amnt": loan_amnt,
        "total_rec_prncp": total_rec_prncp,
        "annual_inc": annual_inc,
        "total_acc": total_acc,
        "funded_amnt": funded_amnt,
        "home_ownership": home_ownership,
        "open_acc": open_acc,
        "verification_status": verification_status,
        "emp_length": emp_length,
    }
for key in result:
    print(f"{key} : {type(result[key])} " )

print("=========== key - values =======")

for key in result:
    print(f"{key} : {result[key] } " )

#==================================Data processing================================================================
# convert data into a type that fit the model
input_data = handle_data(result)

#==================================Predict and display the result=================================================
if not zip_code or not emp_title or not title: 
    st.error("Please fill out all required fields: ZIP Code, Employment Title, and Loan Title.") 
elif not is_valid_zip_code(zip_code): 
    st.error("Please enter a valid ZIP Code (first 3 digits only).")
else:

    if st.button('Predict'):
        print(f" predict in : {input_data.values.reshape(1,-1)}")
        prediction = xgb_model.predict(input_data)
        print(prediction)
        if prediction[0] == 1:
            st.success('Approved')
        else:
            st.error('Not Approved')
            
            # Giải thích quyết định từ chối
            top_features = explain_xgb_prediction(xgb_model, input_data)
            st.markdown("### Explanation for Rejection (top three features that contributes to this rejection)")
            for idx, row in top_features.iterrows():
                feature_name = row['feature']
                explanation = FEATURE_EXPLANATIONS.get(feature_name, "There is no features for this explaination.")
                st.markdown(f"**{idx + 1}. {feature_name}**: {explanation}")
                st.markdown(f"This features effects {row['shap_value']:.2f}.")



# if st.button('Predict'):
#     print(f" predict in : {input_data.values.reshape(1,-1)}")
#     prediction = xgb_model.predict(input_data)
#     print(prediction)
#     if prediction[0] == 1:
#         st.success('Approved')
#     else:
#         st.error('Not Approved')
        
#         # Giải thích quyết định từ chối
#         top_features = explain_xgb_prediction(xgb_model, input_data)
#         st.markdown("### Explanation for Rejection")
#         for idx, row in top_features.iterrows():
#             st.markdown(f"**{idx + 1}. {row['feature']}** contributed to the rejection with an impact of {row['shap_value']:.2f}.")



# if st.button('Predict'):
#     print(f" predict in : {input_data.values.reshape(1,-1)}")
#     #dmatrix_data = xgb.DMatrix(input_data.values, feature_names=input_data.columns)
#     prediction = xgb_model.predict(input_data)
#     print(prediction)
#     st.success(f'{"Approved " if prediction[0] == 1 else "Not Approved"}')

#     # Lời giải thích

#     # Hiển thị lời giải thích
#     st.markdown("### Explanation")
#     st.markdown(explanation)