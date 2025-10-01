from src.load_data import load_and_clean
from src.preprocessing import variance_filter, correlation_filter, stat_filter
from src.feature_selection import fs_corrsf, fs_boruta, fs_rfe_svm, fs_lasso, fs_rf_importance
from src.models import get_models_and_params
from src.evaluation import run_experiments
from src.visualization import plot_results
import pandas as pd
from pathlib import Path

# Paths
base = Path("/workspace/data/validation_data/LungCancer/pp_evita")
path = base / "labeled_radiomics_features.csv"
results_dir = base / "feature_selection_results_modular"
results_dir.mkdir(exist_ok=True)
features_dir = results_dir / "selected_features1000"
features_dir.mkdir(exist_ok=True)

# Load + clean
X, y = load_and_clean(path)

# Preprocess
X = variance_filter(X)
X = correlation_filter(X)
X = stat_filter(X, y)

# Feature selection
methods = {
    "CorrSF": lambda: fs_corrsf(X, y, top_k=30),
    "Boruta": lambda: fs_boruta(X, y),
    "RFE-SVM": lambda: fs_rfe_svm(X, y, n_features=30),
    "L1-LASSO": lambda: fs_lasso(X, y),
    "RF-imp": lambda: fs_rf_importance(X, y, top_k=30),
}
selected_datasets = {}
for name, func in methods.items():
    selected = func()
    selected_datasets[name] = X[selected]
    pd.Series(selected).to_csv(features_dir / f"selected_{name}.csv", index=False)

# Models + params
models, param_grids = get_models_and_params()

# Evaluate
results_df = run_experiments(selected_datasets, y, models, param_grids)
results_df.to_csv(results_dir / "ml_results.csv", index=False)

# Plot
plot_results(results_df, results_dir / "ml_results.png")
