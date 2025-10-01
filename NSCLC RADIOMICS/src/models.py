from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def get_models_and_params():
    models = {
        "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42),
        "Logistic Regression (L1)": LogisticRegression(penalty="l1", solver="saga", max_iter=5000, class_weight="balanced"),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, class_weight="balanced"),
    }
    params = {
        "Random Forest": {
            "clf__n_estimators": [100, 300, 500],
            "clf__max_depth": [None, 10, 20],
            "clf__max_features": ["sqrt", "log2"]
        },
        "Logistic Regression (L1)": {
            "clf__C": [0.01, 0.1, 1, 10]
        },
        "SVM (RBF)": {
            "clf__C": [0.1, 1, 10],
            "clf__gamma": ["scale", "auto"]
        }
    }
    return models, params
