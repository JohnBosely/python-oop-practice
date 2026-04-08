# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_breast_cancer
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from sklearn.preprocessing import StandardScaler
# import pandas as pd

# cancer = load_breast_cancer()
# X = cancer.data
# y = cancer.target

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)


# for c in [0.001, 0.01, 0.1, 10.0, 100.0, 1.0]:
#     for gamma in [0.001, 0.01, 0.1, 1]:
#         sv = SVC(kernel='rbf', C=c, gamma=gamma)
#         # model = SVC(kernel='linear', C=1.0)
#         sv.fit(X_train, y_train)
#         train_acc = sv.score(X_train, y_train)
#         test_acc = sv.score(X_test, y_test)
#         print(f"C: {float(c):<6} Gamma:{float(gamma):.3f} Train: {train_acc:.3f} Test: {test_acc:.3f}")

# best_model = SVC(kernel='rbf', C=10.0, random_state=42, gamma=0.01) 
# best_model.fit(X_train, y_train)
# predictions = best_model.predict(X_test)


# print("ACCURACY:", accuracy_score(y_test, predictions))
# print("CONFUSION MATRIX:\n", confusion_matrix(y_test, predictions))
# print("CLASSIFICATION REPORT:\n", classification_report(y_test, predictions))



from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

def load_and_clean():
    cancer = load_breast_cancer()
    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y = cancer.target
    return X, y
  
def evaluate(best_model, X_test, y_test):
    predictions = best_model.predict(X_test)
    print("\n--- FINAL EVALUATION ---")
    print("ACCURACY:", accuracy_score(y_test, predictions))
    print("CONFUSION MATRIX:\n", confusion_matrix(y_test, predictions))
    print("CLASSIFICATION REPORT:\n", classification_report(y_test, predictions))

def main():
    # Load
    X, y = load_and_clean()
     
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Scale and Tune 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for c in [0.001, 0.01, 0.1, 10.0, 100.0, 1.0]:
        for gamma in [0.001, 0.01, 0.1, 1]:
            sv = SVC(kernel='rbf', C=c, gamma=gamma)
            sv.fit(X_train_scaled, y_train)
            train_acc = sv.score(X_train_scaled, y_train)
            test_acc = sv.score(X_test_scaled, y_test)
            print(f"C: {float(c):<6} Gamma:{float(gamma):.3f} Train: {train_acc:.3f} Test: {test_acc:.3f}")

    # Train best model
    best_model = SVC(kernel='rbf', C=10.0, random_state=42, gamma=0.01) 
    best_model.fit(X_train_scaled, y_train)

    # Evaluate
    evaluate(best_model, X_test_scaled, y_test)

    # Importance Note
    print("\n[Note] plot_importance was skipped because RBF kernels don't support feature importance.")



main()
















