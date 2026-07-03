import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score
)

# LOAD DATASET


data = pd.read_csv("dataset/train.csv")

# FEATURES AND TARGET

X = data.drop(
    ["loan_status", "cb_person_default_on_file"],
    axis=1
)

y = data["loan_status"]

# LABEL ENCODING
home_encoder = LabelEncoder()

intent_encoder = LabelEncoder()

grade_encoder = LabelEncoder()

# ENCODE COLUMNS

X["person_home_ownership"] = home_encoder.fit_transform(
    X["person_home_ownership"]
)

X["loan_intent"] = intent_encoder.fit_transform(
    X["loan_intent"]
)

X["loan_grade"] = grade_encoder.fit_transform(
    X["loan_grade"]
)


# TRAIN TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# RANDOM FOREST MODEL


model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)


# MODEL TRAINING


model.fit(X_train, y_train)

# PROBABILITY PREDICTION


y_prob = model.predict_proba(X_test)[:, 1]


# THRESHOLD TUNING


threshold = 0.42

y_pred = (y_prob >= threshold).astype(int)

# SAVE MODEL

joblib.dump(model, "model/loan_model.pkl")

print("\nModel Saved Successfully")

# =========================================
# SAVE ENCODERS
# =========================================

joblib.dump(
    home_encoder,
    "model/home_encoder.pkl"
)

joblib.dump(
    intent_encoder,
    "model/intent_encoder.pkl"
)

joblib.dump(
    grade_encoder,
    "model/grade_encoder.pkl"
)

print("Encoders Saved Successfully")