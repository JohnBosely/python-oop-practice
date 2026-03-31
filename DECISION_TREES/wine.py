from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load 
wine = load_wine()
X = wine.data
y = wine.target
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target

df.to_csv("wine.csv", index=False)
print("saved")

# Explore
# print(df.head())
# print(df.describe)
# print(df.columns)
# print(df.isnull().sum())

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train
model = DecisionTreeClassifier()
model.fit(X_train, y_train)


predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)
cr = classification_report(y_test, predictions)
print("ACCURACY SCORE: ", accuracy)
print("CONFUSION MATRIX: ", cm)
print("CLASSIFICATION REPORT: ", cr)


model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

plt.figure(figsize=(10, 5))
plot_tree(model,
          feature_names=wine.feature_names,
          class_names=wine.target_names,
          filled=True,
          rounded=True)
plt.show()

for depth in [2, 3, 5, 10, None]:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Depth {str(depth):<6} Train: {train_acc:.3f} Test: {test_acc:.3f}")

for alpha in [0.0, 0.001, 0.01, 0.05, 0.1]:
    model = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Alpha: {alpha:<6} Train: {train_acc:.3f}  Test: {test_acc:.3f}")