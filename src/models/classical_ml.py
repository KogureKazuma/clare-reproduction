"""Eight classical ML models as described in the CLARE paper."""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import lightgbm as lgb
import xgboost as xgb


def build_models() -> dict[str, Pipeline]:
    """Return dict of model name → sklearn Pipeline."""
    scaler = StandardScaler

    models = {
        "GB": Pipeline([
            ("scaler", scaler()),
            ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)),
        ]),
        "LGBM": Pipeline([
            ("scaler", scaler()),
            ("clf", lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)),
        ]),
        "LDA": Pipeline([
            ("scaler", scaler()),
            ("clf", LinearDiscriminantAnalysis()),
        ]),
        "LR": Pipeline([
            ("scaler", scaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "MLP": Pipeline([
            ("scaler", scaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=42)),
        ]),
        "RF": Pipeline([
            ("scaler", scaler()),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]),
        "SVM": Pipeline([
            ("scaler", scaler()),
            ("clf", SVC(kernel="rbf", C=1.0, probability=True, random_state=42)),
        ]),
        "XGBoost": Pipeline([
            ("scaler", scaler()),
            ("clf", xgb.XGBClassifier(n_estimators=200, eval_metric="logloss",
                                       random_state=42, verbosity=0)),
        ]),
    }
    return models
