from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_explore():
    df = pd.read_csv('cs-training.csv')
    print(df.isnull().sum())
    print(df.describe())
    print(df.info())
    print(df.shape)
    # df['RevolvingUtilizationOfUnsecuredLines'].describe()
    
    # 1. Compare Mean vs Median
    print(df[['MonthlyIncome', 'NumberOfDependents']].agg(['mean', 'median']))

    # 2. Check Skewness (if > 1 or < -1, it is highly skewed)
    print(df[['MonthlyIncome', 'NumberOfDependents']].skew())

    # Since there's a skewness of 114, the graph would look useless and towards the left
    # sns.histplot(df['MonthlyIncome'], kde=True)
    # plt.title("Is Income Skewed?")
    # plt.show()

    # Plot to the log-transformed version
    log_income = np.log1p(df['MonthlyIncome'])
    sns.histplot(log_income, kde=True)
    plt.title('Log Transformed Monthly Income')
    plt.show()
    print(df['MonthlyIncome'].skew())
    print(log_income.skew())

    # Missing Indicator
    df['MonthlyIncome_missing'] = df['MonthlyIncome'].isnull().astype(int)

    # Imputebetter with more features
    # Create the imputer
    imputer = KNNImputer(n_neighbors=5)

    # We want to use age and DebtRatio to 'guess' the Income
    columns_to_impute = ['MonthlyIncome', 'age', 'DebtRatio']

    # Fit and transform
    imputed_data = imputer.fit_transform(df[columns_to_impute])

    # Assign the results back to the original dataframe properly
    df[columns_to_impute] = imputed_data
    # Clip Outliers
    df['MonthlyIncome'] = df['MonthlyIncome'].clip(upper=df['MonthlyIncome'].quantile(0.99))

    # Feature Engineering
    df['IncomePerPerson'] = df['MonthlyIncome'] / (df['NumberOfDependents'] + 1)
    # # Calculating Skewness (If > 1, it is highly skewed)
    # print(f"Skewness: {df['MonthlyIncome'].skew()}")


load_and_explore()
        