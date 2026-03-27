from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load 
Cancer = load_breast_cancer()
X = Cancer.data
y = Cancer.target

# Explore it
df = pd.DataFrame(Cancer.data, columns=Cancer.feature_names)
print(df.head())
print(df.describe)
print(df.columns)
print(df.isnull().sum())

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Remind me why and the meaning of the random state)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)                                            
# Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)
cr = classification_report(y_test, predictions)
print("ACCURACY SCORE: ", accuracy)
print("CONFUSION MATRIX: ", cm)
print("CLASSIFICATION REPORT: ", cr)

# Prediction for a single patient
sample_patient = np.array([[17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 
                             0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053,
                             8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587,
                             0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0,
                             0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]])

sample_scaled = scaler.transform(sample_patient)

# Predict
prediction = model.predict(sample_scaled)
probability = model.predict_proba(sample_scaled)

print("Prediction:", "Malignant (Cancer)" if prediction[0] == 0 else "Benign (No Cancer)")
print(f"Probability of Malignant: {probability[0][0]*100:.1f}%")
print(f"Probability of Benign: {probability[0][1]*100:.1f}%")

# Use actual rows from the dataset to test
sample_patient = np.array([Cancer.data[0]])   # known malignant
sample_patient = np.array([Cancer.data[100]]) # try different rows