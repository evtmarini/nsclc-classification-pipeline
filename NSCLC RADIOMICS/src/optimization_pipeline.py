# Import libraries + packages
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib
import optuna
import shap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from imblearn.over_sampling import ADASYN


#  Balance Data
def balance_data(X, y):
    print(" Applying ADASYN oversampling :")
    ada = ADASYN(sampling_strategy="minority", random_state=42, n_neighbors=3)
    X_res, y_res = ada.fit_resample(X, y)
    print(f"[INFO] Dataset balanced: {len(y)} → {len(y_res)} samples")
    return X_res, y_res


# Dimensionality Reduction (PCA)
def apply_pca(X, variance=0.98):
    print(f" Reducing dimensionality (variance={variance}): ")
    pca = PCA(n_components=variance, random_state=42)
    X_pca = pca.fit_transform(X)
    print(f"Reduced {X.shape[1]} → {X_pca.shape[1]} components")
    return pd.DataFrame(X_pca)


# Optuna Optimization
def build_model(trial, model_name):
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 200, 800),
            max_depth=trial.suggest_int("max_depth", 5, 40),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 4),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    elif model_name == "svm":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                C=trial.suggest_float("C", 0.1, 100, log=True),
                gamma=trial.suggest_categorical("gamma", ["scale", "auto", 0.01, 0.1, 1]),
                kernel="rbf",
                probability=True,
                random_state=42
            ))
        ])
    elif model_name == "lgb":
        return LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 200, 800),
            num_leaves=trial.suggest_int("num_leaves", 31, 127),
            max_depth=trial.suggest_int("max_depth", -1, 20),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 20),
            subsample=trial.suggest_float("subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )


def optuna_optimize(X, y, model_name, n_trials=50):
    print(f"\n OPTUNA Optimizing {model_name.upper()} ({n_trials} trials)")

    def objective(trial):
        model = build_model(trial, model_name)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=-1)
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print(f"🏆 {model_name.upper()} Best F1: {study.best_value:.4f}")
    print(f"📦 Best params: {study.best_params}")
    return study.best_params



# Stacking Ensemble
def stacking_ensemble(X, y, rf_params, svm_params, lgb_params):
    print("\n STACKING Building ensemble model...")
    rf = RandomForestClassifier(**rf_params, random_state=42)
    svm = SVC(**svm_params, probability=True, random_state=42)
    lgb = LGBMClassifier(**lgb_params, random_state=42)

    stack = StackingClassifier(
        estimators=[("rf", rf), ("svm", svm), ("lgb", lgb)],
        final_estimator=LogisticRegression(max_iter=500),
        n_jobs=-1
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(stack, X, y, cv=cv, scoring="f1", n_jobs=-1)
    mean_f1 = np.mean(scores)
    print(f" Final Stacking Ensemble F1: {mean_f1:.4f}")
    return stack, mean_f1


# SHAP Analysis
def shap_analysis(model, X, results_dir="data/optuna_results"):
    print("\n SHAP Calculating feature importance: ")
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, show=False)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        path = results_dir / f"shap_summary_{timestamp}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"SAVED SHAP summary -> {path}")
    except Exception as e:
        print(f"WARN SHAP failed: {e}")


# MAIN PIPELINE
def run_full_optimization(X, y, results_dir="data/optuna_results"):
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    # Balance
    X_bal, y_bal = balance_data(X, y)
    # PCA
    X_pca = apply_pca(X_bal, variance=0.98)
    # Optuna tuning
    rf_params = optuna_optimize(X_pca, y_bal, "rf", n_trials=60)
    svm_params = optuna_optimize(X_pca, y_bal, "svm", n_trials=60)
    lgb_params = optuna_optimize(X_pca, y_bal, "lgb", n_trials=60)
    # Stacking
    final_model, f1 = stacking_ensemble(X_pca, y_bal, rf_params, svm_params, lgb_params)
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_path = Path(results_dir) / f"stacking_best_{timestamp}.pkl"
    joblib.dump(final_model, model_path)
    print(f"SAVED Final stacking model -> {model_path}")

    # SHAP
    shap_analysis(final_model, X_pca, results_dir)
    print(f"\n Full optimization completed — Final F1 = {f1:.4f}")
    return f1
