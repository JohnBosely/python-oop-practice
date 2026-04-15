import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_data(df, target):
    # Drop columns with too many missing values (more than 50%)
    threshold = len(df) * 0.5
    df = df.dropna(thresh=threshold, axis=1)

    # Drop columns that are clearly useless
    drop_cols = []
    for col in df.columns:
        if col == target:
            continue
        if df[col].nunique() > 50 and df[col].dtype == object:
            drop_cols.append(col)
        if col.lower() in ["id", "passengerid", "name", "ticket"]:
            drop_cols.append(col)

    df = df.drop(columns=drop_cols, errors='ignore')

    # Fill missing values — check numeric properly
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Encode text columns to numbers
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            df[col] = le.fit_transform(df[col].astype(str))

    return df