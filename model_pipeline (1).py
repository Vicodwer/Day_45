# Week 08 Tuesday Assignment
# Deep Learning + Data Cleaning

import numpy as np
import pandas as pd

# ----------------------------
# Sub-step 1: Data Audit
# ----------------------------
def audit_data(df):
    report = {}
    for col in df.columns:
        report[col] = {
            "missing": df[col].isnull().sum(),
            "unique": df[col].nunique(),
            "dtype": str(df[col].dtype)
        }
    return report


# ----------------------------
# Sub-step 2: Data Cleaning
# ----------------------------
def clean_data(df):
    df = df.copy()
    df = df.drop_duplicates()

    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


# ----------------------------
# Sub-step 3: Neural Network
# ----------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def initialize_weights(input_size, hidden_size, output_size):
    return {
        "W1": np.random.randn(input_size, hidden_size) * 0.01,
        "W2": np.random.randn(hidden_size, output_size) * 0.01
    }


def forward(X, weights):
    Z1 = X.dot(weights["W1"])
    A1 = sigmoid(Z1)
    Z2 = A1.dot(weights["W2"])
    A2 = sigmoid(Z2)
    return A1, A2


# ----------------------------
# Sub-step 4: Training
# ----------------------------
def train(X, y, epochs=1000, lr=0.01):
    weights = initialize_weights(X.shape[1], 16, 1)

    for i in range(epochs):
        A1, A2 = forward(X, weights)
        loss = np.mean((A2 - y)**2)

        if i % 100 == 0:
            print(f"Epoch {i}, Loss: {loss}")

    return weights


# ----------------------------
# Sub-step 5: Evaluation
# ----------------------------
def evaluate(y_true, y_pred):
    return np.mean(y_true == y_pred)


if __name__ == "__main__":
    print("Run each sub-step in notebook for full pipeline.")
