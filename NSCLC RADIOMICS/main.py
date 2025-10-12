import pandas as pd
from src.load_data import load_and_clean
from src.preprocessing import variance_filter, correlation_filter, stat_filter
from src.feature_selection import (
    fs_mrmr, fs_relieff, fs_corrsf, fs_ses,
    fs_boruta, fs_rfe_svm, fs_genetic,
    fs_lasso, fs_hsic_lasso, fs_rf_importance
)
from src.models import get_models_and_params
from src.evaluation import run_experiments
from src.visualization import plot_results
from pathlib import Path
from src.optuna_report import optuna_report
from src.optimization_pipeline import run_full_optimization
from src.f1_boost_config import get_f1_boost_config
#from src.evaluation_optuna import optuna_tune_model, optuna_stacking


# Paths

base = Path("data")
path = base / "labeled_radiomics_features.csv"
results_dir = base / "feature_selection_results"
features_dir = results_dir / "selected_features"
results_dir.mkdir(exist_ok=True)
features_dir.mkdir(exist_ok=True)


# Load dataset
print("Loading dataset")
X, y = load_and_clean(path)


# Preprocessing
print("\n Preprocessing:")
X = variance_filter(X)
X = correlation_filter(X)
X = stat_filter(X, y)
print(f" After preprocessing: {X.shape[1]} features retained.")


# Feature Selection
print("\n Feature Selection methods running...")

methods = {
    "mRMR": lambda: fs_mrmr(X, y, top_k=30),
    "ReliefF": lambda: fs_relieff(X, y, top_k=30),
    "CorrSF": lambda: fs_corrsf(X, y, top_k=30),
    "SES": lambda: fs_ses(X, y),
    "Boruta": lambda: fs_boruta(X, y),
    "RFE-SVM": lambda: fs_rfe_svm(X, y, n_features=30),
    "Genetic": lambda: fs_genetic(X, y, top_k=30),
    "L1-LASSO": lambda: fs_lasso(X, y),
    "HSIC-LASSO": lambda: fs_hsic_lasso(X, y, top_k=30),
    "RF-imp": lambda: fs_rf_importance(X, y, top_k=30)
}


selected_datasets = {}
for name, func in methods.items():
    try:
        selected = func()
        selected_datasets[name] = X[selected]
        pd.Series(selected).to_csv(features_dir / f"selected_{name}.csv", index=False)
        print(f" {name}: {len(selected)} features saved.")
    except Exception as e:
        print(f" {name} failed: {e}")


# Models
print("\n Loading models and hyperparameter grids:")
models, param_grids = get_models_and_params()
print(f" Loaded {len(models)} models.")


# Halving Random Search for ALL FS + Models
print("\n Halving Random Search on all feature sets and models: ")
results_df = run_experiments(selected_datasets, y, models, param_grids, cv=5)


# Save & Plot
results_path = results_dir / "halving_results_all.csv"
results_df.to_csv(results_path, index=False)
plot_results(results_df, results_dir / "halving_results_all.png")
print("\n Halving Search completed for all models and FS methods.")
print(f" Results saved to: {results_path}")


print("\n Optuna Bayesian Optimization: ")
X_sel = selected_datasets["Boruta"]  # or "CorrSF"

print("\n Optuna Bayesian Optimization...")
X_sel = selected_datasets["mRMR"]  # or other FS method

rf_f1, rf_params = optuna_report(X_sel, y, model_name="rf", n_trials=30)
svm_f1, svm_params = optuna_report(X_sel, y, model_name="svm", n_trials=30)
lgb_f1, lgb_params = optuna_report(X_sel, y, model_name="lgb", n_trials=30)

print("\n Stacking Ensemble: ")


print("\n Full Optimization Pipeline (ADASYN + PCA + Optuna + Stacking + SHAP)")
X_sel = selected_datasets["mRMR"]  # or other FS method
final_f1 = run_full_optimization(X_sel, y)
print(f" Final optimized model F1-score: {final_f1:.4f}")


# Activate F1-Boost Mode
cfg = get_f1_boost_config(mode="f1_boost")
methods = cfg["feature_selection"](X, y, mode="f1_boost")
models, param_grids = cfg["models_and_params"]()
