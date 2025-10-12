# Import libraries + packages
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

def load_and_clean(path, id_col="case_id", target_col="label", min_class_size=10):
    df = pd.read_csv(path)
    print(f" Loading Dataset: {df.shape}")

    # Labels
    y = LabelEncoder().fit_transform(df[target_col])
    X = df.drop(columns=[c for c in [id_col, target_col] if c in df.columns])
    X = X.select_dtypes(include=[np.number])

    # Class distribution check
    unique, counts = np.unique(y, return_counts=True)
    print("\n Class distribution before filtering:")
    for cls, count in zip(unique, counts):
        print(f"    Class {cls}: {count} samples")

    # Find classes with really few samples
    small_classes = [cls for cls, c in zip(unique, counts) if c < min_class_size]
    if len(small_classes) > 0:
        print(f"\n Found small classes with <{min_class_size} samples: {small_classes}")
        mask = ~np.isin(y, small_classes)
        X = X[mask]
        y = y[mask]
        print(f" Filtered dataset shape: {X.shape}, remaining classes: {np.unique(y)}")
    else:
        print(" All classes have sufficient samples.")

    # Clean Nan/Inf
    X = X.replace([np.inf, -np.inf], np.nan)
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns)

    print(f"\n Cleaned dataset ready: {X.shape[0]} samples × {X.shape[1]} features")
    return X, y
