# evaluation.py
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, precision_score, recall_score


def run_experiments(selected_datasets, y, models, param_grids, cv=4):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scoring = {
        "f1_weighted": "f1_weighted",
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, average="weighted", zero_division=0),
        "recall": make_scorer(recall_score, average="weighted", zero_division=0),
    }
    results = []
    for fs_name, X_sel in selected_datasets.items():
        for model_name, clf in models.items():
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
            grid = GridSearchCV(pipe, param_grids[model_name], scoring=scoring,
                                refit="f1_weighted", cv=skf, n_jobs=-1)
            grid.fit(X_sel, y)

            best_idx = grid.best_index_
            results.append({
                "FS_method": fs_name,
                "Classifier": model_name,
                "F1_score": grid.cv_results_["mean_test_f1_weighted"][best_idx],
                "Accuracy": grid.cv_results_["mean_test_accuracy"][best_idx],
                "Precision": grid.cv_results_["mean_test_precision"][best_idx],
                "Recall": grid.cv_results_["mean_test_recall"][best_idx],
                "Best_params": grid.best_params_,
            })
    return pd.DataFrame(results)
