from sklearn.linear_model import LinearRegression
import numpy as np

# dataset
X = np.array([[2], [4], [6], [8]])
y = np.array([40, 50, 65, 80])

# create model
model = LinearRegression()

# train model
model.fit(X, y)

# prediction
prediction = model.predict([[10]])

print(prediction)

