from sklearn.datasets import load_wine, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# CLASSIFICATION -> WINE
wine = load_wine()

X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.2, random_state=42)
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42) # Meaning of N_estimators and why 100
rf_clf.fit(X_train, y_train)
print("Random Forest Wine accuracy:", rf_clf.score(X_test, y_test))
print("Decision Tree Wine was:       0.944")

# This is to test for the best number of trees needed.
# It was observed that 50 is the sweet spot as no other improvement occured
# for n in [10, 50, 100, 200, 500]:
#     rf_clf = RandomForestClassifier(n_estimators=n, random_state=42) # Meaning of N_estimators and why 100
#     rf_clf.fit(X_train, y_train)
#     score = rf_clf.score(X_test, y_test)
#     print(f"Trees: {n:<6} Accuracy: {score:.4f}")

# REGRESSION -> DIABETES
diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=42)
for depth in [3, 5, 10, None]:
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=depth, random_state=42)
    rf_reg.fit(X_train, y_train)
    print("Random Forest Diabetes R2:", rf_reg.score(X_test,y_test))
print("Decision Tree Diabetes was:  0.334")
print("Linear Regression was:       0.4526")



rf_best = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
rf_best.fit(X_train, y_train)

importance = pd.Series(rf_best.feature_importances_, index=diabetes.feature_names)
importance.sort_values().plot(kind='barh')
plt.title("Feature Importance - Random Forest")
plt.show()



