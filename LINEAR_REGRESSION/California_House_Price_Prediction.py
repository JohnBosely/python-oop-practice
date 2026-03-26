from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Load
housing = fetch_california_housing()
X = housing.data
y = housing.target

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train 
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Predict
predictions = model.predict(X_test)

#5. Measure
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)
print("Mean Squared Error: ", mse)
print("RMSE: ", np.sqrt(mse))
print("R2: ", r2)

plt.figure(figsize=(10,6)) #what does this mean
plt.scatter(y_test, predictions, alpha=0.3) #what does this mean
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
plt.show()
#also explain those terms means squared, r2 and so on explain why they are useful, i tend to forget often

# This is used to show the actual prices and the predicted prices
plt.figure(figsize=(10,6))
plt.scatter(X_test[:, 0], y_test, alpha=0.3, label='Actual')
plt.scatter(X_test[:, 0], predictions, alpha=0.3, label='Predicted', color='red')
plt.xlabel("MedInc (feature 0)")
plt.ylabel("Price")
plt.legend()
plt.title("What the model actually learned")
plt.show()

# This is used find the most important features in the model, however it is not advised to 
# use it cos it doesnt tel the full story about a dataset
my_house = np.array([[5.0, 20.0, 6.0, 1.0, 800.0, 3.0, 37.5, -122.0]])
predicted_price = model.predict(my_house)
print(f"Predicted price: ${predicted_price[0] * 100000:.2f}")

feature_names = housing.feature_names
coefficients = model.coef_

for feature, coef in zip(feature_names, coefficients):
    print(f"{feature}: {coef:.4f}")



