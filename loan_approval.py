# %%
"""
Load Loan Approval Classification Dataset from Kaggle.

Utilizing this dataset we will be classifying loan applicants into two
categories: approved or denied. The dataset includes features 
about loan applicants such as their age, income, education, and loan intent. 

The dataset can be found on the Kaggle Datasets website here: 
https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data/data
"""

import os
import joblib
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import KFold, GridSearchCV

# Scale the features using MinMaxScaler (between 0 and 1).
def transform_data(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                    scale_cols: list[str]) -> tuple[NDArray, NDArray]:
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train[scale_cols])
    X_test_scaled = scaler.transform(X_test[scale_cols])
    return X_train_scaled, X_test_scaled

# Add new features to the dataset
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df["emp_exp_ratio"] = df["person_emp_exp"] / (df["person_age"] + 1)
    df["credit_score_per_year"] = (
        df["credit_score"] / (df["cb_person_cred_hist_length"] + 1)
        )
    df["high_risk_flag"] = (
        (df["loan_int_rate"] > df["loan_int_rate"].median()) &
        (df["loan_percent_income"] > df["loan_percent_income"].median())
    ).astype(int)
    df["age_bin"] = pd.cut(df["person_age"], 
                            bins=[18, 25, 35, 50, 65, 100], labels=False)
    df["edu_income_interaction"] = (
        df["person_education_Master"] + df["person_education_Doctorate"]
    ) * df["person_income"]
    return df

def get_rule_for_sample(model, feature_names, sample):
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

def main() -> None:
    df = pd.read_csv("loan_data.csv")

    # Drop unrealistic outliers (age > 100)
    df = df[df["person_age"] <= 100].reset_index(drop=True)

    # List of columns to scale between 0-1 for better test results.
    scale_cols = [
        "person_age", "person_income", "person_emp_exp", "loan_amnt", 
        "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length", 
        "credit_score"
    ]

    # Convert categorical columns into numerical format
    df["person_gender"] = df["person_gender"].map({"female": 1, "male": 0})
    df["previous_loan_defaults_on_file"] = (
        df["previous_loan_defaults_on_file"].map({"Yes": 1, "No": 0})
    )
    df = pd.get_dummies(df, columns=["person_education", "person_home_ownership", 
                        "loan_intent"], dtype=int)

    # Apply feature engineering
    df = feature_engineering(df)

    # Split dataset into features and target
    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    # Split data into train and test subsets (80% train - 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42)

    # Determine which columns to scale
    scale_cols = [col for col in X_train.columns if X_train[col].dtype in 
                    [np.float64, np.int64]]
    X_train_scaled, X_test_scaled = transform_data(X_train, X_test, scale_cols)

    # Define model and hyperparameters
    model = DecisionTreeClassifier(random_state=42)
    parameters = {
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"]
    }

    # 4-Fold Cross Validation setup
    kf = KFold(n_splits=4, shuffle=True, random_state=42)

    # Baseline performance
    base_scores = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring="accuracy")
    print(f"Baseline mean accuracy: {base_scores.mean():.3f}")

    # Hyperparameter tuning
    grid = GridSearchCV(model, parameters, cv=kf, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train_scaled, y_train)

    print("Best parameters:", grid.best_params_)
    print(f"Tuned mean accuracy: {grid.best_score_:.3f}")

    # Evaluate on test data
    best_model = grid.best_estimator_
    test_acc = accuracy_score(y_test, best_model.predict(X_test_scaled))
    print(f"Test accuracy: {test_acc:.3f}")

    # Create a DataFrame with predictions + interpretability column
    preds = best_model.predict(X_test_scaled)
    interpretability = [
        get_rule_for_sample(best_model, scale_cols, x)
        for x in X_test_scaled
    ]

    # Build results dataframe
    results_df = X_test.copy()
    results_df["true_label"] = y_test.values
    results_df["prediction"] = preds
    results_df["interpretability"] = interpretability

    # Save to CSV
    results_df.to_csv("results.csv", index=False)

    print("Saved detailed prediction results to results.csv")

    # Save model
    MODEL_DIR = "models"
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "decision_tree_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"Saved Decision Tree model to {model_path}")
    # Save the best model using joblib (recommended by scikit-learn)
    # joblib is more efficient than pickle for sklearn models with large numpy arrays.
    # The model file can be loaded later using: joblib.load('model_filename.joblib')


if __name__ == '__main__':
    main()

# %%
