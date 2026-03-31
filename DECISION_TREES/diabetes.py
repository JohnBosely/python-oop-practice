from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df["target"] = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

for depth in [2,3,4,5,10,None]:
    model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    print(f"Depth: {str(depth):<6} Train: {train_r2:.3f} Test: {test_r2:.3f}")

for alpha in [0.0, 0.001, 0.01, 0.05, 0.1]:
    model = DecisionTreeRegressor(ccp_alpha=alpha, random_state=42)
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Alpha: {alpha:<6} Train: {train_acc:.3f}  Test: {test_acc:.3f}")