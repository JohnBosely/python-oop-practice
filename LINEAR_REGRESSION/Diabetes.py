from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


diabetes = load_diabetes()

# load Dataset
X = diabetes.data
y = diabetes.target
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

# Tells you exactly how many empty cells are in each column
# print(df.isnull().sum())


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 )

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)


# Predict
predictions = model.predict(X_test)

# Measure
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("R2: ", r2)

df["target"] = y

correlations = df.corr()["target"].sort_values(ascending=False)
print(correlations)

# Visual representation
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlations")
plt.show()


# I dropped some values because there were weak relationships, after plotting them without the dropped variables, r2 dropped
# This means that the dataset is limited so removing variables with low relationship wont work
# Diabetes is determined by more variables like genetics, diet, excercise habits, medication, stresslevels
X_new = df.drop(["target", "sex", "s2"], axis=1)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_new, y, test_size=0.2, random_state=42)
model2 = LinearRegression()
model.fit(X_train2, y_train2)

prediction2 = model.predict(X_test2)
print("Original R2:", 0.4526)
print("New R2:", r2_score(y_test2, prediction2))


# print(df.head())
# print(df.columns)
# print(df.describe)