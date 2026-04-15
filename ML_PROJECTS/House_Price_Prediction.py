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
    model = RandomForestRegressor(n_estimators=100, max_depth=20)
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

def plot_importance(rankings):
    plt.figure(figsize=(10, 6))
    rankings.head(10).plot(kind='barh', color='skyblue')
    plt.title("Top 10 Most Influential Features")
    plt.gca().invert_yaxis() # Put the best at the top
    plt.xlabel("Importance Score")
    plt.show()
plot_importance(rankings)


# --- STEP 1: Finalize Training ---
top_20_features = rankings.head(20).index.tolist()
X_train_final = X_final[top_20_features]

model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5 )
model.fit(X_train_final, y) # Training on the WHOLE train.csv

# --- STEP 2: Process the Separate Test Data ---
def load_and_prep_test(url, features_to_keep):
# Add 'r' before the quotes to prevent path errors
    test_df = pd.read_csv(url)
    
    # Fill missing values (Important!)
    test_df = test_df.fillna(test_df.median(numeric_only=True))
    
    # Filter for the SAME 20 columns you used for training
    X_test = test_df[features_to_keep]
    
    # If test.csv has SalePrice, extract it; if not, return None for y
    y_test = test_df['SalePrice'] if 'SalePrice' in test_df.columns else None
    return X_test, y_test

# --- STEP 3: Evaluate or Predict ---
test_path = r'C:\Users\USER\Documents\PYTHON OOP PROJECT\ML_PROJECTS\test.csv'
X_test_final, y_test_actual = load_and_prep_test(test_path, top_20_features)

if y_test_actual is not None:
    # Use your evaluate function if you have the answers
    evaluate(model, X_test_final, y_test_actual)
else:
    # Just get the predictions if it's for a submission
    predictions = model.predict(X_test_final)
    print("Predictions generated for test.csv!")

# --- STEP 4: Save for Submission ---
if y_test_actual is None:
    # We need the original test IDs for the submission file
    original_test = pd.read_csv(test_path)
    
    submission = pd.DataFrame({
        "Id": original_test["Id"],
        "SalePrice": predictions
    })
    
    submission.to_csv("my_submission.csv", index=False)
    print("Success! Your predictions are saved in 'my_submission.csv'")


