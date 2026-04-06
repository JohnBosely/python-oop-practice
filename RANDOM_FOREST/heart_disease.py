# ------ IMPORTS ------ #
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------ LOAD AND CLEAN ------ #
def load_and_clean():
    url = "https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv"
    df = pd.read_csv(url)
    return df

# ── 3. EVALUATE ──
def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    print("ACCURACY:", accuracy_score(y_test, predictions))
    print("CONFUSION MATRIX:\n", confusion_matrix(y_test, predictions))
    print("CLASSIFICATION REPORT:\n", classification_report(y_test, predictions))

# ── 4. FEATURE IMPORTANCE ──
def plot_importance(model, columns):
    importance = pd.Series(model.feature_importances_, index=columns)
    importance.sort_values().plot(kind='barh')
    plt.title("Feature Importance")
    plt.show()

# ── 5. MAIN — where everything runs ──
def main():
    # Load
    df = load_and_clean()
    X = df.drop("target", axis=1)
    y = df["target"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Tune
    for n, depth in [(100, 5), (200, 5), (100, 10), (200, 10)]:
        rf = RandomForestClassifier(n_estimators=n, max_depth=depth, random_state=42)
        rf.fit(X_train, y_train)
        print(f"Trees: {n} Depth: {depth} Accuracy: {rf.score(X_test, y_test):.4f}")

    # Train best model
    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    evaluate(model, X_test, y_test)

    # Visualise
    plot_importance(model, X.columns)

main()
















