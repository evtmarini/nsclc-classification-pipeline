import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.preprocessing import StandardScaler


def fs_corrsf(X, y, top_k=30, corr_max=0.85):
    selector = SelectKBest(score_func=f_classif, k="all").fit(X, y)
    scores = pd.Series(selector.scores_, index=X.columns).fillna(0.0).sort_values(ascending=False)
    selected = []
    for feat in scores.index:
        if len(selected) >= min(top_k, X.shape[1]):
            break
        if selected and X[selected].corrwith(X[feat]).abs().max() >= corr_max:
            continue
        selected.append(feat)
    return selected


def fs_boruta(X, y):
    rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5, random_state=42)
    bor = BorutaPy(estimator=rf, n_estimators="auto", random_state=42)
    bor.fit(X.values, y)
    return X.columns[bor.support_].tolist()


def fs_rfe_svm(X, y, n_features=30):
    estimator = SVC(kernel="linear")
    rfe = RFE(estimator=estimator, n_features_to_select=min(n_features, X.shape[1]), step=0.1)
    rfe.fit(StandardScaler().fit_transform(X), y)
    return X.columns[rfe.support_].tolist()


def fs_lasso(X, y):
    lasso = LogisticRegression(penalty="l1", solver="saga", max_iter=5000, class_weight="balanced")
    lasso.fit(StandardScaler().fit_transform(X), y)
    coef_mean = np.mean(np.abs(lasso.coef_), axis=0)
    return X.columns[coef_mean > 1e-6].tolist()


def fs_rf_importance(X, y, top_k=30):
    rf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced")
    rf.fit(X, y)
    return pd.Series(rf.feature_importances_, index=X.columns).nlargest(top_k).index.tolist()
