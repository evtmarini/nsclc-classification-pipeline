#Import libraries + packages
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from boruta import BorutaPy
from skrebate import ReliefF

# Filter methods
def fs_mrmr(X, y, top_k=30):
    print(f"Running mRMR (fallback) for {top_k} features...")
    mi = mutual_info_classif(X, y, random_state=42)
    scores = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    selected = []
    for feat in scores.index:
        if len(selected) >= top_k:
            break
        if selected and X[selected].corrwith(X[feat]).abs().max() > 0.85:
            continue
        selected.append(feat)
    print(f" mRMR selected {len(selected)} features.")
    return selected


def fs_relieff(X, y, top_k=30):
    print(f" Running ReliefF for {top_k} features...")
    X_scaled = StandardScaler().fit_transform(X)
    relief = ReliefF(n_neighbors=20, n_features_to_select=top_k, n_jobs=-1)
    relief.fit(X_scaled, y)
    feats = X.columns[relief.top_features_[:top_k]].tolist()
    print(f"ReliefF selected {len(feats)} features.")
    return feats


def fs_corrsf(X, y, top_k=40, corr_max=0.9):
    print(f"Running CorrSF (top {top_k})...")
    selector = SelectKBest(score_func=f_classif, k="all").fit(X, y)
    scores = pd.Series(selector.scores_, index=X.columns).fillna(0).sort_values(ascending=False)
    selected = []
    for feat in scores.index:
        if len(selected) >= top_k:
            break
        if selected and X[selected].corrwith(X[feat]).abs().max() >= corr_max:
            continue
        selected.append(feat)
    print(f" CorrSF selected {len(selected)} features.")
    return selected


def fs_ses(X, y, alpha=0.1):
    print("Running SES (Kruskal/Mann–Whitney filtering)...")
    from scipy.stats import kruskal, mannwhitneyu
    classes = np.unique(y)
    selected = []
    for col in X.columns:
        groups = [X[y == cls][col] for cls in classes]
        stat, p = (mannwhitneyu(*groups) if len(classes) == 2 else kruskal(*groups))
        if p < alpha:
            selected.append(col)
    print(f" SES retained {len(selected)} significant features.")
    return selected


# WRAPPER METHODS
def fs_boruta(X, y):
    print("Running Boruta feature selection...")
    rf = RandomForestClassifier(
        n_jobs=-1,
        class_weight="balanced",
        max_depth=7,          
        random_state=42
    )
    bor = BorutaPy(
        estimator=rf,
        n_estimators="auto",
        perc=80,              
        random_state=42
    )
    bor.fit(X.values, y)
    selected = X.columns[bor.support_].tolist()
    print(f" Boruta selected {len(selected)} features.")
    return selected


def fs_rfe_svm(X, y, n_features=30):
    print(f" Running RFE with linear SVM (target {n_features})...")
    estimator = SVC(kernel="linear", random_state=42)
    rfe = RFE(estimator=estimator, n_features_to_select=min(n_features, X.shape[1]), step=0.1)
    rfe.fit(StandardScaler().fit_transform(X), y)
    feats = X.columns[rfe.support_].tolist()
    print(f"RFE-SVM selected {len(feats)} features.")
    return feats


def fs_genetic(X, y, top_k=20, generations=8, pop_size=25):
    print(f"Running Genetic Algorithm for {generations} generations...")
    rng = np.random.default_rng(42)
    n_features = X.shape[1]
    population = rng.integers(0, 2, size=(pop_size, n_features))
    best_subset, best_score = None, 0

    for g in range(generations):
        scores = []
        for chrom in population:
            subset = X.columns[chrom == 1]
            if len(subset) == 0:
                scores.append(0)
                continue
            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            f1s = []
            for train, val in cv.split(X, y):
                clf.fit(X.iloc[train][subset], y.iloc[train])
                preds = clf.predict(X.iloc[val][subset])
                f1s.append(f1_score(y.iloc[val], preds, average="weighted"))
            scores.append(np.mean(f1s))
        best_idx = np.argmax(scores)
        if scores[best_idx] > best_score:
            best_score = scores[best_idx]
            best_subset = population[best_idx]
        parents = population[np.argsort(scores)[-5:]]
        children = []
        for _ in range(pop_size):
            p1, p2 = parents[rng.integers(0, 5)], parents[rng.integers(0, 5)]
            child = np.where(rng.random(n_features) < 0.5, p1, p2)
            mut_mask = rng.random(n_features) < 0.05
            child[mut_mask] ^= 1
            children.append(child)
        population = np.array(children)

    selected = X.columns[best_subset == 1].tolist()
    print(f" GA selected {len(selected)} features.")
    return selected[:top_k]


# EMBEDDED METHODS
def fs_lasso(X, y):
    print("Running tuned L1-LASSO selection: ")
    X_scaled = StandardScaler().fit_transform(X)
    lasso = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=2,                    
        max_iter=10000,
        class_weight="balanced",
        random_state=42
    )
    lasso.fit(X_scaled, y)
    coef_mean = np.mean(np.abs(lasso.coef_), axis=0)
    feats = X.columns[coef_mean > 1e-5].tolist()
    print(f"LASSO retained {len(feats)} features.")
    return feats


def fs_hsic_lasso(X, y, top_k=20):
    try:
        from pyHSICLasso import HSICLasso
        print("Running HSIC-Lasso...")
        hsic = HSICLasso()
        hsic.input(np.array(X), np.array(y))
        hsic.classification(top_k)
        feats = [X.columns[i] for i in hsic.get_index()]
        print(f"HSIC-Lasso selected {len(feats)} features.")
        return feats
    except ImportError:
        print("pyHSICLasso not installed. Skipping HSIC-Lasso.")
        return []


def fs_rf_importance(X, y, top_k=30):
    print("Running Random Forest importance selection...")
    rf = RandomForestClassifier(
        n_estimators=500, 
        random_state=42, 
        n_jobs=-1, 
        class_weight="balanced"
    )
    rf.fit(X, y)
    feats = pd.Series(rf.feature_importances_, index=X.columns).nlargest(top_k).index.tolist()
    print(f"RF-importance selected {len(feats)} features.")
    return feats
