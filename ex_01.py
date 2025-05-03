from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
from math import log

annual_income = np.array([30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]).reshape(-1, 1)
refund = np.array([0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1])

model = LogisticRegression(solver='liblinear')
model.fit(annual_income, refund)

b0 = model.intercept_[0]
b1 = model.coef_[0][0]

print(f"Intercept (β₀): {model.intercept_[0]:.2f}")
print(f"Coefficient (β₁): {model.coef_[0][0]:.2f}")

equation = f"P(pass) = 1 / (1 + e^-({model.intercept_[0]:.2f} + {model.coef_[0][0]:.2f} × K USD))"
print(f"Logistic equation: {equation}")


def get_threshold(probability):
    logit = log(probability / (1 - probability))
    return (logit - b0) / b1


decision_boundary = get_threshold(0.5)
print(f"Decision boundary: {decision_boundary:.2f}K USD")

print("\nQuestion_2:")
print("Return on loan if annual is 58K USD: ")
print("Will probably return the loan,\nloan APPROVED!") if decision_boundary < 58 \
    else print("Will likely not return the loan,\nloan REJECTED!")


print("\nQuestion_3:")
decision_boundary = get_threshold(0.75)
print(f"Decision boundary: {decision_boundary:.2f}K USD")
print(f"If the client has an annual income of {decision_boundary:.2f}K USD he will most certainly return the loan")