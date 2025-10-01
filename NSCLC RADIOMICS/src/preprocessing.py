import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import mannwhitneyu, kruskal


def variance_filter(X, threshold=0.01):
    vt = VarianceThreshold(threshold=threshold)
    return pd.DataFrame(vt.fit_transform(X), columns=X.columns[vt.get_support()])


def correlation_filter(X, threshold=0.85):
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = []
    for column in upper.columns:
        high_corr = upper[column][upper[column] > threshold].index.tolist()
        for hc in high_corr:
            mean_corr_col = corr[column].mean()
            mean_corr_hc = corr[hc].mean()
            drop = column if mean_corr_col > mean_corr_hc else hc
            if drop not in to_drop:
                to_drop.append(drop)
    return X.drop(columns=to_drop)


def stat_filter(X, y, alpha=0.1):
    classes = np.unique(y)
    selected = []
    for col in X.columns:
        groups = [X[y == cls][col] for cls in classes]
        stat, p = (mannwhitneyu(*groups) if len(classes) == 2 else kruskal(*groups))
        if p < alpha:
            selected.append(col)
    return X[selected]
