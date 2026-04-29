from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_explore():
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    df = df.drop(columns=['customerID'])
        # 1. Force conversion (This creates the 'True' Nulls)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # 2. Now check for nulls - you'll see 11!
    print(f"Nulls after conversion: {df['TotalCharges'].isnull().sum()}")
    # 3. Fill them
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
# This should be around (7043, 31), not 6000+!
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
    df['Dependents'] = df['Dependents'].map({'Yes': 1, 'No': 0})
    df['PhoneService'] = df['PhoneService'].map({'Yes': 1, 'No': 0})
    df_final = pd.get_dummies(df, drop_first=True)
    print(df.head())
    print(df.info())
    print(df.describe())
    print(f"Final shape: {df_final.shape}") 
    print(df['Churn'].value_counts(normalize=True))
    X = df_final.drop('Churn', axis=1)
    y = df_final['Churn']
    sns.countplot(x='Churn', data=df)
    plt.title('Distribution of Churn')
    plt.show()
        # Assuming your target column is named 'Churn'
    counts = df_final['Churn'].value_counts()

    # Index 0 is usually 'Stayed', Index 1 is 'Churned'
    # Formula: Negative Count / Positive Count
    scale_pos_weight = counts[0] / counts[1]

    print(f"Recommended scale_pos_weight: {scale_pos_weight:.2f}")

    return df_final


def evaluate(model,X_test, y_test):
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

def plot_importance(model, columns):
    importance = pd.Series(model.feature_importances_, index=columns)
    importance.sort_values().plot(kind='barh')
    plt.title("Feature Importance")
    plt.show()

def main():
    # Load
    df = load_and_explore()
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Tune
    for n, depth in [(100, 5), (200, 5), (100, 10), (200, 10)]:
        rf = RandomForestClassifier(n_estimators=n, max_depth=depth, random_state=42, class_weight='balanced')
        rf.fit(X_train, y_train)
        print(f"Trees: {n} Depth: {depth} Accuracy: {rf.score(X_test, y_test):.4f}")

    # Train best model
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # Evaluate
    evaluate(model, X_test, y_test)

    # # Visualise
    plot_importance(model, X.columns)

main()



