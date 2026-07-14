"""Champion and challenger model development with frozen chronological splits."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

NUMERIC_FEATURES = [
    "fico_score",
    "ltv",
    "dti",
    "loan_amount",
    "rate_spread",
    "prior_delinquency",
    "unemployment_rate",
    "hpi_yoy_change",
]
CATEGORICAL_FEATURES = ["loan_purpose", "occupancy_status", "state"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "default_12m"


def _preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
    )


def champion_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocess", _preprocessor()),
            ("model", LogisticRegression(C=1.0, max_iter=1_000, random_state=20260713)),
        ]
    )


def challenger_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocess", _preprocessor()),
            (
                "model",
                DecisionTreeClassifier(
                    max_depth=4, min_samples_leaf=150, class_weight=None, random_state=20260713
                ),
            ),
        ]
    )


def fit_models(df: pd.DataFrame, model_dir: Path | None = None) -> tuple[Pipeline, Pipeline]:
    train = df.loc[df["split"] == "train"]
    champion = champion_pipeline().fit(train[FEATURES], train[TARGET])
    challenger = challenger_pipeline().fit(train[FEATURES], train[TARGET])
    if model_dir is not None:
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(champion, model_dir / "champion_logistic.joblib")
        joblib.dump(challenger, model_dir / "challenger_tree.joblib")
    return champion, challenger


def coefficient_table(champion: Pipeline) -> pd.DataFrame:
    feature_names = champion.named_steps["preprocess"].get_feature_names_out()
    coefficients = champion.named_steps["model"].coef_[0]
    return pd.DataFrame(
        {
            "feature": feature_names,
            "model_coefficient": coefficients,
            "odds_ratio_per_model_unit": np.exp(coefficients),
        }
    ).sort_values("model_coefficient", key=lambda s: s.abs(), ascending=False)
