# Includes Halving Random Search

#Import libraries + packages
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import StratifiedKFold, HalvingRandomSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, make_scorer
from imblearn.over_sampling import ADASYN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

#execute halving search for each classifier
#adasyn balancing
def run_experiments(selected_datasets, y, models, param_grids, cv=2):
    results = []
    os.makedirs("data", exist_ok=True)

    for fs_name, X_sel in selected_datasets.items():
        print(f"\n[INFO] Running Halving Search for feature set: {fs_name} ({X_sel.shape[1]} features)")
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        # Data balancing with adasyn
        print(" Balancing data with ADASYN: ")
        ada = ADASYN(sampling_strategy="auto", random_state=42, n_neighbors=3)
        X_bal, y_bal = ada.fit_resample(X_sel, y)
        print(f" Balanced dataset: {len(y)} → {len(y_bal)} samples")

        for model_name, clf in models.items():
            print(f"\n Evaluating: {model_name} ")
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", clf)
            ])

            search = HalvingRandomSearchCV(
                estimator=pipe,
                param_distributions=param_grids.get(model_name, {}),
                scoring=make_scorer(f1_score, average="weighted"),
                cv=skf,
                factor=4,                      # faster halving
                min_resources='smallest',       
                random_state=42,
                n_jobs=-1,
                verbose=1,
                error_score=0.0
            )

            search.fit(X_bal, y_bal)
            best_score = search.best_score_
            best_params = search.best_params_

            print(f" {model_name}: F1 = {best_score:.4f}")

            results.append({
                "FS_method": fs_name,
                "Classifier": model_name,
                "F1_score": best_score,
                "Best_params": best_params
            })

    # Results
    results_df = pd.DataFrame(results)
    csv_path = "data/halving_results.csv"
    results_df.to_csv(csv_path, index=False)

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.barh(results_df["Classifier"], results_df["F1_score"], color="skyblue")
    plt.xlabel("Weighted F1-score")
    plt.title(f"Halving Random Search — {fs_name}")
    plt.tight_layout()
    plt.savefig("data/halving_results.png", dpi=300)
    plt.close()

    print(f"\n Results saved to {csv_path}")
    return results_df
