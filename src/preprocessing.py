import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_data(df):
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled
