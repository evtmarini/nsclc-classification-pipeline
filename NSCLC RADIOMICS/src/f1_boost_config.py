# High-F1 Configuration for Radiomics Pipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import PowerTransformer, StandardScaler
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import optuna


# FEATURE SELECTION CONFIG
def get_fs_methods(X, y, mode="f1_boost"):
    from src.feature_selection import fs_corrsf, fs_boruta, fs_relieff, fs_hsic_lasso

    if mode == "fast":
        methods = {
            "CorrSF": lambda: fs_corrsf(X, y, top_k=20),
            "Boruta": lambda: fs_boruta(X, y)
        }
    else:  # f1_boost
        methods = {
            "CorrSF": lambda: fs_corrsf(X, y, top_k=30),
            "Boruta": lambda: fs_boruta(X, y),
            "ReliefF": lambda: fs_relieff(X, y, top_k=30),
            "HSIC-LASSO": lambda: fs_hsic_lasso(X, y, top_k=30)
        }

    return methods


# PREPROCESSING CONFIG
def transform_features(X):
    print(" Applying PowerTransformer + StandardScaler: ")
    X_pt = PowerTransformer().fit_transform(X)
    return pd.DataFrame(StandardScaler().fit_transform(X_pt), columns=X.columns)



# SAMPLING CONFIG
def balance_data(X, y):
    print(" Applying Borderline-SMOTE: ")
    smote = BorderlineSMOTE(kind='borderline-1', random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print(f" Samples balanced: {len(y)} → {len(y_res)}")
    return X_res, y_res


# DIMENSIONALITY REDUCTION
def apply_pca(X, variance=0.95):
    print(f" PCA Reducing variance to {variance}...")
    pca = PCA(n_components=variance, random_state=42)
    X_pca = pca.fit_transform(X)
    print(f"[INFO] {X.shape[1]} → {X_pca.shape[1]} components retained.")
    return pd.DataFrame(X_pca)



# MODELS & PARAM GRIDS
def get_models_and_params():
    rf = RandomForestClassifier(class_weight="balanced", random_state=42)
    svm = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)

    stacking_model = StackingClassifier(
        estimators=[("rf", rf), ("svm", svm)],
        final_estimator=LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            num_leaves=15,
            random_state=42
        ),
        passthrough=True,
        n_jobs=-1
    )

    models = {
        "Random Forest": rf,
        "SVM (RBF)": svm,
        "Stacking Ensemble (RF+SVM+LGBM)": stacking_model
    }

    params = {
        "Random Forest": {
            "clf__n_estimators": [300, 600],
            "clf__max_depth": [10, 20],
            "clf__max_features": ["sqrt"],
            "clf__min_samples_leaf": [1, 2]
        },
        "SVM (RBF)": {
            "clf__C": [5, 10, 20],
            "clf__gamma": ["scale", "auto"]
        },
        "Stacking Ensemble (RF+SVM+LGBM)": {
            "clf__final_estimator__n_estimators": [100, 200],
            "clf__final_estimator__learning_rate": [0.05],
            "clf__final_estimator__max_depth": [2, 3]
        }
    }

    return models, params


# OPTUNA CONFIG
def create_optuna_study():
    return optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )


# FAST/F1-BOOST CONFIG WRAPPER
def get_f1_boost_config(mode="f1_boost"):
    config = {
        "mode": mode,
        "feature_selection": get_fs_methods,
        "transform_features": transform_features,
        "balance_data": balance_data,
        "apply_pca": apply_pca,
        "models_and_params": get_models_and_params,
        "optuna_study": create_optuna_study
    }
    return config


if __name__ == "__main__":
    cfg = get_f1_boost_config("f1_boost")
    print(" Loaded F1-Boost configuration.")
