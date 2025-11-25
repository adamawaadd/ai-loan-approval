import streamlit as st
import pandas as pd
import numpy as np
from numpy.typing import NDArray
import joblib
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler

# -----------------------------
#  Import feature engineering
# -----------------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df["emp_exp_ratio"] = df["person_emp_exp"] / (df["person_age"] + 1)
    df["credit_score_per_year"] = (
        df["credit_score"] / (df["cb_person_cred_hist_length"] + 1)
    )
    df["high_risk_flag"] = (
        (df["loan_int_rate"] > df["loan_int_rate"].median()) &
        (df["loan_percent_income"] > df["loan_percent_income"].median())
    ).astype(int)
    df["age_bin"] = pd.cut(
        df["person_age"], bins=[18, 25, 35, 50, 65, 100], labels=False
    )
    df["edu_income_interaction"] = (
        df["person_education_Master"] + df["person_education_Doctorate"]
    ) * df["person_income"]
    return df


# -----------------------------
# Load Model + Rebuild Scaler
# -----------------------------
@st.cache_data
def load_model_and_scaler():
    script_dir = Path(__file__).resolve().parent

    # Load trained model
    model_path = script_dir / "models" / "decision_tree_model.joblib"
    model = joblib.load(model_path)

    # Load training data to recreate scaling + columns
    df = pd.read_csv(script_dir / "loan_data.csv")

    df = df[df["person_age"] <= 100].reset_index(drop=True)

    # Convert categorical fields
    df["person_gender"] = df["person_gender"].map({"female": 1, "male": 0})
    df["previous_loan_defaults_on_file"] = (
        df["previous_loan_defaults_on_file"].map({"Yes": 1, "No": 0})
    )
    df = pd.get_dummies(
        df,
        columns=["person_education", "person_home_ownership", "loan_intent"],
        dtype=int,
    )

    df = feature_engineering(df)

    X = df.drop(columns=["loan_status"])

    numeric_cols = [c for c in X.columns if X[c].dtype in [np.int64, np.float64]]

    scaler = MinMaxScaler()
    scaler.fit(X[numeric_cols])

    return model, scaler, numeric_cols, X.columns.tolist()



# -----------------------------
# Interpretability Function
# -----------------------------
def get_rule_for_sample(model, feature_names, sample):
    from sklearn.tree import _tree

    tree_ = model.tree_
    feature = tree_.feature
    threshold = tree_.threshold

    node = 0
    rules = []

    while feature[node] != _tree.TREE_UNDEFINED:
        feat_name = feature_names[feature[node]]
        thresh = threshold[node]

        if sample[feature[node]] <= thresh:
            rules.append(f"{feat_name} <= {thresh:.3f}")
            node = tree_.children_left[node]
        else:
            rules.append(f"{feat_name} > {thresh:.3f}")
            node = tree_.children_right[node]

    return " → ".join(rules)



# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.title("📊 Loan Approval Prediction App")
    st.write("Enter applicant info below to predict loan approval.")

    model, scaler, numeric_cols, column_order = load_model_and_scaler()

    with st.form("loan_form"):
        st.subheader("Applicant Information")
        person_age = st.number_input("Age", 18, 100, 30)
        person_income = st.number_input("Annual Income", 0, 500000, 60000)
        person_emp_exp = st.number_input("Years Employed", 0, 40, 2)
        credit_score = st.number_input("Credit Score", 300, 850, 700)
        cb_length = st.number_input("Credit History Length (years)", 0, 40, 5)
        gender = st.selectbox("Gender", ["male", "female"])

        st.subheader("Loan Details")
        loan_amnt = st.number_input("Loan Amount", 100, 1000000, 15000)
        loan_int_rate = st.number_input("Interest Rate (%)", 1.0, 40.0, 12.0)
        loan_percent_income = st.number_input(
            "Loan Percent of Income", 0.0, 1.0, 0.15
        )

        loan_intent = st.selectbox(
            "Loan Intent",
            ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
        )

        home = st.selectbox(
            "Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"]
        )

        edu = st.selectbox(
            "Education Level", ["High School", "Bachelor", "Master", "Doctorate", "Other"]
        )

        default_history = st.selectbox(
            "Previous Loan Defaults?", ["No", "Yes"]
        )

        submitted = st.form_submit_button("Predict Approval")

    if submitted:
        # Build a single-row dataframe
        user = {
            "person_age": person_age,
            "person_income": person_income,
            "person_emp_exp": person_emp_exp,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_cred_hist_length": cb_length,
            "credit_score": credit_score,
            "person_gender": 1 if gender == "female" else 0,
            "previous_loan_defaults_on_file": 1 if default_history == "Yes" else 0,
        }

        # Convert categorical to one-hot
        user_df = pd.DataFrame([user])

        user_df["person_education_" + edu] = 1
        user_df["person_home_ownership_" + home] = 1
        user_df["loan_intent_" + loan_intent] = 1

        # Add missing dummy columns
        for col in column_order:
            if col not in user_df.columns:
                user_df[col] = 0

        # Reorder columns
        user_df = user_df[column_order]

        # Feature engineering
        user_df = feature_engineering(user_df)

        # Scale numeric columns only
        user_scaled = user_df.copy()
        user_scaled[numeric_cols] = scaler.transform(user_scaled[numeric_cols])

        pred = model.predict(user_scaled.to_numpy())[0]
        prob = model.predict_proba(user_scaled.to_numpy())[0][1]


        if pred == 1:
            st.success(f"Loan Approved (Probability: {prob:.2%})")
        else:
            st.error(f"Loan Denied (Probability: {prob:.2%})")

        st.subheader("Decision Path for This Applicant")
        rule_text = get_rule_for_sample(model, numeric_cols, user_scaled[numeric_cols].iloc[0].values)
        st.code(rule_text)



if __name__ == "__main__":
    main()
