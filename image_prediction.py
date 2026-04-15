import pandas as pd
import os
import time
from datetime import datetime
from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt

# digits classifier
digits = datasets.load_digits()
clf = svm.SVC(gamma=0.001, C=100)
x, y = digits.data[:-1], digits.target[:-1]
clf.fit(x, y)
print("Predictions:", clf.predict(digits.data[-8].reshape(1, -1)))
print("Actual:", digits.target[-8])

plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()

