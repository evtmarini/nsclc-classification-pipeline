import optuna
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np


#  Helper: Generic model builder
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


#  Main Optuna + SHAP + Gini Importance Report
def optuna_report(X, y, model_name, n_trials=60, results_dir="data/optuna_results"):
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n Running Optuna optimization for {model_name.upper()} ({n_trials} trials): ")

    # Objective
    def objective(trial):
        model = build_model(trial, model_name)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scorer = make_scorer(f1_score, average="weighted")
        scores = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=-1)
        return np.mean(scores)

    # Run study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial
    print(f" Best F1: {best_trial.value:.4f}")
    print(f" Best params: {best_trial.params}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Save all trial results
    df = study.trials_dataframe()
    df.to_csv(results_dir / f"{model_name}_optuna_{timestamp}.csv", index=False)

    # Train best model
    best_model = build_model(best_trial, model_name)
    best_model.fit(X, y)
    joblib.dump(best_model, results_dir / f"{model_name}_best_model_{timestamp}.pkl")

    
    # FEATURE IMPORTANCE: GINI + SHAP
    print("\n Generating feature importance (Gini + SHAP): ")

    feature_names = X.columns if hasattr(X, "columns") else [f"feat_{i}" for i in range(X.shape[1])]
    shap_path_prefix = results_dir / f"{model_name}_{timestamp}"

    try:
        # Gini Importance
        if hasattr(best_model, "feature_importances_"):
            gini_imp = pd.Series(best_model.feature_importances_, index=feature_names).sort_values(ascending=False)
        elif hasattr(best_model, "named_steps") and hasattr(best_model.named_steps["clf"], "feature_importances_"):
            gini_imp = pd.Series(best_model.named_steps["clf"].feature_importances_, index=feature_names).sort_values(ascending=False)
        else:
            gini_imp = None

        if gini_imp is not None:
            plt.figure(figsize=(8, 6))
            gini_imp.head(20).plot(kind="barh")
            plt.title(f"{model_name.upper()} – Top 20 Gini Importances")
            plt.xlabel("Gini Importance")
            plt.tight_layout()
            plt.savefig(f"{shap_path_prefix}_gini.png", dpi=300, bbox_inches="tight")
            plt.close()
            print(f" Gini importance plot saved.")

        # SHAP Importance 
        if model_name in ["rf", "lgb"]:
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X)
        else:
            explainer = shap.Explainer(best_model.predict_proba, X)
            shap_values = explainer(X)

        shap.summary_plot(shap_values, X, show=False)
        plt.title(f"SHAP Summary – {model_name.upper()}")
        plt.tight_layout()
        plt.savefig(f"{shap_path_prefix}_shap_summary.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f" SHAP summary plot saved.")

        # Gini vs SHAP Comparison
        if gini_imp is not None:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            shap_imp = pd.Series(mean_abs_shap, index=feature_names)
            common_feats = gini_imp.index.intersection(shap_imp.index)
            corr = gini_imp[common_feats].corr(shap_imp[common_feats])
            print(f" Gini vs SHAP importance correlation: {corr:.3f}")

            plt.figure(figsize=(6, 6))
            plt.scatter(gini_imp[common_feats], shap_imp[common_feats], alpha=0.7)
            plt.xlabel("Gini Importance")
            plt.ylabel("Mean SHAP value")
            plt.title(f"Gini vs SHAP Correlation (r={corr:.2f})")
            plt.tight_layout()
            plt.savefig(f"{shap_path_prefix}_gini_shap_correlation.png", dpi=300)
            plt.close()
            print(f"✅ Gini vs SHAP correlation plot saved.")

    except Exception as e:
        print(f" SHAP or Gini importance failed: {e}")

    print(f" Results saved to {results_dir}")
    return best_trial.value, best_trial.params
