# Import libraries + packages
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def get_models_and_params():

    # Random Forest
    rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=42
    )

    # Optimized SVM (RBF) 
    svm = Pipeline([
        ("scaler", StandardScaler()),  
        ("pca", PCA(n_components=0.9, random_state=42)), 
        ("clf", SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42
        ))
    ])

    # Stacking Ensemble (RF + SVM + Gradient Boosting)
    stacking_model = StackingClassifier(
        estimators=[
            ("rf", rf),
            ("svm", svm)
        ],
        final_estimator=GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ),
        passthrough=True,
        n_jobs=-1
    )

    # Soft Voting Ensemble
    soft_voting = VotingClassifier(
        estimators=[
            ("rf", rf),
            ("svm", svm)
        ],
        voting="soft",
        weights=[1, 1],
        n_jobs=-1
    )

    # Models
    models = {
        "Random Forest": rf,
        "SVM (RBF)": svm,
        "Stacking Ensemble (RF+SVM+GB)": stacking_model,
        "Soft Voting (RF+SVM)": soft_voting
    }

    # Hyperparameter grid
    params = {
        "Random Forest": {
            "clf__n_estimators": [300, 600, 1000],
            "clf__max_depth": [10, 20, None],
            "clf__min_samples_split": [2, 5],
            "clf__min_samples_leaf": [1, 2],
            "clf__max_features": ["sqrt", "log2"],
        },


        "SVM (RBF)": {
            "clf__clf__C": [0.1, 1, 10, 50, 100],
            "clf__clf__gamma": [1e-4, 1e-3, 0.01, 0.1, "scale"],
            "clf__pca__n_components": [0.85, 0.9, 0.95],
        },

        "Stacking Ensemble (RF+SVM+GB)": {
            "clf__final_estimator__n_estimators": [100, 200, 300],
            "clf__final_estimator__learning_rate": [0.03, 0.05, 0.1],
            "clf__final_estimator__max_depth": [2, 3, 4],
        },

        "Soft Voting (RF+SVM)": {
            "clf__weights": [(1, 1), (2, 1), (1, 2)]
        }
    }

    return models, params


if __name__ == "__main__":
    models, params = get_models_and_params()
    print("Available models:")
    for name in models.keys():
        print(f" - {name}")
    print("\n Parameters for SVM:")
    print(params["SVM (RBF)"])
