import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import shap
import warnings

warnings.filterwarnings('ignore')

# 1. DATA LOADING FUNCTION
def load_and_explore():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
    cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigree", "Age", "Outcome"]
    df = pd.read_csv(url, names=cols)
    
    # Clean "impossible" zeros
    cols_to_fix = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[cols_to_fix] = df[cols_to_fix].replace(0, np.nan)
    for col in cols_to_fix:
        df[col] = df[col].fillna(df[col].median())
    
    print(f"Dataset Loaded: {df.shape[0]} patients, {df.shape[1]-1} medical features.")
    return df

# 2. THE MAIN PIPELINE
def main():
    # --- A. Data Prep ---
    df = load_and_explore()
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- B. Model Audition ---
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True),
        "XGBoost": XGBClassifier(eval_metric='logloss')
    }

    print("\n--- Model Audition Results ---")
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        score = model.score(X_test_scaled, y_test)
        print(f"{name}: {score:.4f} Accuracy")

    # --- C. SHAP Explainability (Using the winner: Random Forest) ---
    best_model = models["Random Forest"]
    
    # We use TreeExplainer for Random Forest
    explainer = shap.TreeExplainer(best_model)
    
    # check_additivity=False handles the floating-point math errors
    shap_values = explainer.shap_values(X_test_scaled, check_additivity=False)

    # Handle the "List vs Array" version difference
    if isinstance(shap_values, list):
        # Older SHAP/RandomForest returns a list [class_0, class_1]
        shap_values_to_plot = shap_values[1]
    elif len(shap_values.shape) == 3:
        # Newer SHAP returns a 3D array (patients, features, classes)
        shap_values_to_plot = shap_values[:, :, 1]
    else:
        # Standard 2D array (XGBoost often does this)
        shap_values_to_plot = shap_values

    # --- D. Visualizations ---
    print("\nGenerating SHAP Summary Plot...")
    # Summary plot handles the data shape automatically most of the time
    shap.summary_plot(shap_values_to_plot, X_test_scaled, feature_names=X.columns)

    print("\nExplaining prediction for Patient #1 using Waterfall Plot...")
    # To use the waterfall plot, we create an 'Explanation' object
    # This combines the base value, the shap values, and the feature names into one package
    
    # We take the data for the first patient [0]
    exp = shap.Explanation(
        values=shap_values_to_plot[0], 
        base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
        data=X_test_scaled[0],
        feature_names=X.columns.tolist()
    )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(exp)
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()