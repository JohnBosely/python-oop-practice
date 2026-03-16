import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("salary_data.csv")

# print(dataset.head())

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=0 )

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# print("Predicted salaries:")
# print(y_pred)
# print(X_test)

plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, model.predict(X_train), color="blue")

plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")

plt.savefig("salary_vs_experience_train.png")
plt.show()


