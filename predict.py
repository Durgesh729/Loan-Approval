import joblib
import pandas as pd

# =========================================
# LOAD MODEL
# =========================================


home_encoder = joblib.load(
    "model/home_encoder.pkl"
)

intent_encoder = joblib.load(
    "model/intent_encoder.pkl"
)

grade_encoder = joblib.load(
    "model/grade_encoder.pkl"
)


model = joblib.load("model/loan_model.pkl")

# =========================================
# NEW CUSTOMER INPUT
# =========================================

new_data = {

    "id": 11111,

    "person_age": 28,

    "person_income": 65000,

    "person_home_ownership": 3,

    "person_emp_length": 5,

    "loan_intent": 4,

    "loan_grade": 1,

    "loan_amnt": 12000,

    "loan_int_rate": 11.5,

    "loan_percent_income": 0.18,

    "cb_person_cred_hist_length": 4
}

# =========================================
# CONVERT TO DATAFRAME
# =========================================

input_data = pd.DataFrame([new_data])

# =========================================
# PREDICTION
# =========================================

prediction = model.predict(input_data)

# =========================================
# DISPLAY RESULT
# =========================================

if prediction[0] == 1:

    print("Loan Approved")

else:

    print("Loan Rejected")