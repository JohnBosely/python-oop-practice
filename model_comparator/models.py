from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd

def run_models(df, target, problem_type):
    X = df.drop(target, axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if problem_type == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
            "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
            "SVM":                 SVC(kernel='rbf', C=10, gamma=0.01)
        }
    else:
        models = {
            "Linear Regression":   LinearRegression(),
            "Decision Tree":       DecisionTreeRegressor(max_depth=5, random_state=42),
            "Random Forest":       RandomForestRegressor(n_estimators=100, random_state=42),
            "SVM":                 SVR(kernel='rbf', C=10, gamma=0.01)
        }

    results = []
    for name, model in models.items():
        # SVM needs scaled data
        if "SVM" in name or "Logistic" in name:
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

        if problem_type == "classification":
            score = round(accuracy_score(y_test, predictions) * 100, 2)
            metric = "Accuracy %"
        else:
            score = round(r2_score(y_test, predictions), 4)
            metric = "R2 Score"

        results.append({
            "Model": name,
            metric: score
        })

    # Sort by score
    results = sorted(results, key=lambda x: list(x.values())[1], reverse=True)
    return results