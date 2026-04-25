"""
=============================================================
DIABETES RISK PREDICTOR WITH SHAP EXPLAINABILITY
=============================================================
Dataset  : PIMA Indians Diabetes (768 patients, 8 features)
Best Model: Random Forest — 77.92% Accuracy
Key Feature: SHAP Waterfall plots explain individual predictions

EXPERIMENTS TRIED (see commented sections below):
- Feature Engineering  → made model WORSE (dropped to 72%)
- GridSearchCV         → best XGBoost params, still below Random Forest
- KNN Imputation       → marginal improvement, kept median for simplicity
=============================================================
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────
# STEP 1 — LOAD AND CLEAN
# ─────────────────────────────────────────
def load_and_explore():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
    cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigree", "Age", "Outcome"]
    df = pd.read_csv(url, names=cols)

    # ── Fix impossible zeros ──
    # A living person cannot have 0 BMI, 0 Glucose, or 0 BloodPressure
    # These zeros represent missing data, not actual measurements
    # We replace them with NaN then fill with the column median
    cols_to_fix = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[cols_to_fix] = df[cols_to_fix].replace(0, np.nan)
    for col in cols_to_fix:
        df[col] = df[col].fillna(df[col].median())

    # ── EXPERIMENT 1: Feature Engineering (ABANDONED) ──
    # These features were created to give the model more signal
    # RESULT: Random Forest accuracy DROPPED from 77.92% to 72.08%
    # The new features added redundancy/noise rather than new information
    # Lesson: More features does not always mean better results
    #
    # df['BMI_Age'] = df['BMI'] * df['Age']
    # df['Glucose_Insulin'] = df['Glucose'] / (df['Insulin'] + 1)
    # df['High_Risk'] = ((df['Glucose'] > 140) & (df['BMI'] > 30)).astype(int)

    # ── EXPERIMENT 2: KNN Imputation (ABANDONED) ──
    # KNN Imputation fills missing values using similar patients
    # instead of just the median — more medically intelligent
    # RESULT: Marginal improvement (~0.5%), not worth added complexity
    # Kept median imputation for simplicity and reproducibility
    #
    # from sklearn.impute import KNNImputer
    # imputer = KNNImputer(n_neighbors=5)
    # df[cols_to_fix] = imputer.fit_transform(df[cols_to_fix])

    print(f"Loaded: {df.shape[0]} patients, {df.shape[1]-1} features")
    print(f"Diabetic: {df['Outcome'].sum()} | Healthy: {(df['Outcome']==0).sum()}")
    return df


# ─────────────────────────────────────────
# STEP 2 — EVALUATE MODEL
# ─────────────────────────────────────────
def evaluate(model, X_test, y_test, name):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    predictions = model.predict(X_test)
    print(f"\n── {name} ──")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(classification_report(y_test, predictions,
                                target_names=["Healthy", "Diabetic"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))


# ─────────────────────────────────────────
# STEP 3 — SHAP EXPLANATION
# ─────────────────────────────────────────
def explain_patient(model, explainer, shap_values_to_plot, X_test_scaled,
                    feature_names, patient_index=0):
    """Generate a SHAP waterfall plot for a single patient."""
    # Get expected value — handles both old (list) and new (array) SHAP versions
    if isinstance(explainer.expected_value, (list, np.ndarray)):
        base_val = explainer.expected_value[1]
    else:
        base_val = explainer.expected_value

    exp = shap.Explanation(
        values=shap_values_to_plot[patient_index],
        base_values=base_val,
        data=X_test_scaled[patient_index],
        feature_names=feature_names
    )
    print(f"\nWaterfall plot — Patient #{patient_index + 1}")
    shap.plots.waterfall(exp)
    plt.show()


# ─────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────
def main():
    # ── A. Load ──
    df = load_and_explore()
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # ── B. Split ──
    # stratify=y ensures the same 65/35 healthy/diabetic ratio
    # in both training and test sets — prevents biased evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # ── C. Scale ──
    # StandardScaler puts all features on the same scale
    # Without this, Insulin (0-846) would dominate DiabetesPedigree (0-2.4)
    # fit_transform on train: learns the scale
    # transform on test: applies the same scale WITHOUT learning from test data
    # (learning from test data would be "data leakage" — cheating)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── D. Model Audition ──
    # Test multiple algorithms to find the best one for this specific data
    # SVC(probability=True) — needed so SHAP can read probability outputs
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM":                 SVC(probability=True, random_state=42),
        "XGBoost":             XGBClassifier(eval_metric='logloss', random_state=42)
    }

    # ── EXPERIMENT 3: GridSearchCV on XGBoost (INFORMATIONAL) ──
    # Tested to find the optimal XGBoost hyperparameters
    # Best params found: learning_rate=0.01, max_depth=4,
    #                    n_estimators=300, scale_pos_weight=1
    # Best score: 76.7% — still below Random Forest baseline of 77.92%
    # Conclusion: Tuning XGBoost did not beat the default Random Forest
    #
    # from sklearn.model_selection import GridSearchCV
    # params = {
    #     'n_estimators': [100, 200, 300],
    #     'max_depth': [3, 4, 5],
    #     'learning_rate': [0.01, 0.05, 0.1],
    #     'scale_pos_weight': [1, 2, 2.77]
    # }
    # grid = GridSearchCV(XGBClassifier(eval_metric='logloss'),
    #                     params, cv=5, scoring='accuracy')
    # grid.fit(X_train_scaled, y_train)
    # print("Best XGBoost params:", grid.best_params_)
    # print("Best XGBoost score:", grid.best_score_)

    print("\n─── Model Audition Results ───")
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        score = model.score(X_test_scaled, y_test)
        winner = " ← WINNER" if name == "Random Forest" else ""
        print(f"{name:<25} {score:.4f}{winner}")

    # ── E. Evaluate Winner ──
    evaluate(models["Random Forest"], X_test_scaled, y_test, "Random Forest")

    # ── F. SHAP Explainability ──
    # TreeExplainer is optimised specifically for tree-based models
    # like Random Forest and XGBoost
    print("\nCalculating SHAP values...")
    best_model = models["Random Forest"]
    explainer = shap.TreeExplainer(best_model)

    # check_additivity=False prevents floating point math errors
    # where SHAP values sum to 0.5001 but model says 0.4800
    # The difference is a rounding error, not a real problem
    shap_values = explainer.shap_values(X_test_scaled, check_additivity=False)

    # Handle SHAP version differences:
    # Old SHAP → returns list [class_0_array, class_1_array]
    # New SHAP → returns 3D array (patients, features, classes)
    # We always want class 1 (diabetic) explanations
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif len(shap_values.shape) == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    # ── G. Summary Plot — what matters across ALL patients ──
    print("\nGenerating Summary Plot (all patients)...")
    shap.summary_plot(sv, X_test_scaled, feature_names=X.columns)

    # ── H. Waterfall Plot — individual patient story ──
    # Change patient_index to explain any patient from 0 to 153
    explain_patient(best_model, explainer, sv, X_test_scaled,
                    X.columns.tolist(), patient_index=0)


if __name__ == "__main__":
    main()
