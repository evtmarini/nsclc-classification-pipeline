import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


def load_and_clean(path, id_col="case_id", target_col="label"):
    df = pd.read_csv(path)
    print(f" Loading Dataset: {df.shape}")

    # Labels
    y = LabelEncoder().fit_transform(df[target_col])
    X = df.drop(columns=[c for c in [id_col, target_col] if c in df.columns])
    X = X.select_dtypes(include=[np.number])

    # Clean NaN/Inf
    X = X.replace([np.inf, -np.inf], np.nan)
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns)

    return X, y
