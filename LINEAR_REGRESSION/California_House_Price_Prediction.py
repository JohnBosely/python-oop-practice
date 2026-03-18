import numpy as np 
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
X = housing.data
y = housing.target

dataset = pd.DataFrame(X, columns=housing.feature_names)
dataset["SalesPrice"] = y #salesprice
# print(dataset.head())
# print(dataset.shape)
# print(dataset.info)
# print(dataset.describe)
# print(dataset.isnull().sum())

# sns.pairplot(dataset.sample(100), height=2.5) This is used to limit the data used to plot
# sns.pairplot(dataset, height=2.5)
# plt.tight_layout()
# sns.histplot(dataset["SalesPrice"])
# sns.kdeplot(dataset["SalesPrice"])
# print("Skewness: %f" % dataset['SalesPrice'].skew()) # 0.977763 apparently skewness and kurtosis are used to get ouliers
# print("Kurtosis: %f" % dataset['SalesPrice'].kurt()) # 0.327870

# fig, ax = plt.subplots()
# ax.scatter(dataset["MedInc"],dataset["SalesPrice"])
# plt.ylabel('SalesPrice', fontsize=13)
# plt.xlabel('MedInc', fontsize=13)
# plt.show()

# dataset.to_excel("CALIFORNIA.xlsx", index=False) This is used to turn a dataset into an excel file

from scipy import stats
from scipy.stats import norm, skew #for some statistics

sns.displot(dataset['SalesPrice'] , fit=norm);

(mu, sigma) = norm.fit(dataset['SalesPrice'])
print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f})'.format(mu, sigma)],
           loc='best')
plt.ylabel('Frequency')
plt.xlabel('Sales distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(dataset['SalesPrice'], plot=plt)
plt.show()

