import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("Advertising_Budget_and_Sales.csv")

print(dataset.head())

print(dataset.info())
print(dataset.isnull().sum())
print(dataset.describe())
