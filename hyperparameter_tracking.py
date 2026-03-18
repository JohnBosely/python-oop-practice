import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

# Load the dataset
digits = datasets.load_digits()
x, y = digits.data[:-1], digits.target[:-1]  # ← add this back

for gamma in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]:  # ← removed 0.0
    clf = svm.SVC(gamma=gamma, C=100)
    clf.fit(x, y)
    prediction = clf.predict(digits.data[-1].reshape(1, -1))
    actual = digits.target[-1]
    correct = prediction[0] == actual
    print(f"Gamma: {gamma:<8} Predicted: {prediction[0]}  Actual: {actual}  Correct: {correct}")