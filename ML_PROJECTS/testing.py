
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# <----- LOAD AND EXPLORE DATASET -----> #
def load_and_explore():
    df = pd.read_csv("train.csv")
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])
    # 1. DROP columns that are text/objects 
    df = df.select_dtypes(include=[np.number])
    # 2. Fill empty spaces
    df = df.fillna(df.median())
    print(df.isnull().sum())
    print(df.describe())
    print(df.info())
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']
    return X, y

# <----- REMOVE CONSTATNT COLUMNS -----> #
def remove_constant_features (df):
    selector = VarianceThreshold(threshold=0)
    selector.fit(df)
    return df[df.columns[selector.get_support()]]

# <----- REMOVE HIGHLY CORRELATED COLUMNS -----> #
def remove_high_correlation(df, treshold=0.95):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > treshold)]
    return df.drop(columns=to_drop)

# <----- RANK BY IMPORTANCE -----> #
def get_features_importance(X, y):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns)
    return importances.sort_values(ascending=False)

X, y = load_and_explore()
X_filtered = remove_constant_features(X)
X_final = remove_high_correlation(X_filtered)

# Get and print the rankings
rankings = get_features_importance(X_final, y)
print("Top 30 Most Important Columns:")
print(rankings.head(30))

def evaluate(model, X, y):
    predictions = model.predict(X)
    r2 = r2_score(y, predictions)
    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    
    # 3. Print results
    print(f"--- Evaluation Results ---")
    print(f"R2 Score: {r2:.4f}")
    print(f"Mean Absolute Error: ${mae:,.2f}")
    print(f"Root Mean Squared Error: ${rmse:,.2f}")

# --- STEP 1: Split and Train ---
top_20_features = rankings.head(20).index.tolist()
X_selected = X_final[top_20_features]

# 1. Split the training data (80% for learning, 20% for testing the model)
X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# 2. Initialize and Train on the 80% chunk
# Note: max_depth=5 might be a bit shallow/simple for this data, but good for preventing overfitting!
model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=20)
model.fit(X_train, y_train)

# 3. Check the Accuracy on the 20% chunk it hasn't seen yet
print("\n--- INTERNAL VALIDATION (80/20 Split) ---")
evaluate(model, X_val, y_val)
