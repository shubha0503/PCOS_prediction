"""Reproducible model comparison and Extra Trees tuning for the PCOS dataset.

Run from the repository root: python train_model.py
The script uses a stratified 80/20 holdout split and 5-fold CV on training data.
"""

from __future__ import annotations

import json
import argparse
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "pcos_cleaned.csv"
TARGET = "PCOS (Y/N)"
RANDOM_STATE = 42


def main(save_artifacts: bool = True) -> None:
    data = pd.read_csv(DATA_PATH)
    if TARGET not in data:
        raise ValueError(f"Expected target column {TARGET!r} in {DATA_PATH}")

    # Patient File No. is an identifier and Unnamed columns are spreadsheet artifacts.
    y = pd.to_numeric(data[TARGET], errors="raise").astype(int)
    X = data.drop(columns=[TARGET], errors="ignore")
    X = X.drop(columns=[c for c in X.columns if c.lower().startswith("unnamed") or "file no" in c.lower()])
    X = X.apply(pd.to_numeric, errors="coerce")
    if y.nunique() != 2 or not set(y.unique()).issubset({0, 1}):
        raise ValueError("Target must contain binary labels 0 and 1")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    candidates = {
        "Extra Trees": ExtraTreesClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE, class_weight="balanced"),
        "KNN": KNeighborsClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        "Naive Bayes": GaussianNB(),
    }

    results = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        results[name] = {
            "test_accuracy": float(accuracy_score(y_test, pred)),
            "test_precision": float(precision_score(y_test, pred, zero_division=0)),
            "test_recall": float(recall_score(y_test, pred, zero_division=0)),
            "test_f1": float(f1_score(y_test, pred, zero_division=0)),
            "test_roc_auc": float(roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])),
        }

    et_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", ExtraTreesClassifier(random_state=RANDOM_STATE, class_weight="balanced")),
    ])
    search = GridSearchCV(
        et_pipeline,
        {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 8, 16],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
            "model__max_features": ["sqrt", 0.8],
        },
        scoring="accuracy", cv=cv, n_jobs=-1, refit=True,
    )
    search.fit(X_train, y_train)

    # Keep the app's existing artifact contract: a fitted transform and estimator.
    transform = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    X_train_transformed = transform.fit_transform(X_train)
    X_test_transformed = transform.transform(X_test)
    tuned_parameters = {key.removeprefix("model__"): value for key, value in search.best_params_.items()}
    final_model = ExtraTreesClassifier(random_state=RANDOM_STATE, class_weight="balanced", **tuned_parameters)
    final_model.fit(X_train_transformed, y_train)
    final_pred = final_model.predict(X_test_transformed)
    final_prob = final_model.predict_proba(X_test_transformed)[:, 1]
    # Keep all reported tuned metrics tied to the exact estimator saved for inference.
    results["Tuned Extra Trees"] = {
        "test_accuracy": float(accuracy_score(y_test, final_pred)),
        "test_precision": float(precision_score(y_test, final_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, final_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, final_pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, final_prob)),
        "confusion_matrix": confusion_matrix(y_test, final_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y_test, final_pred, output_dict=True, zero_division=0),
    }

    if save_artifacts:
        joblib.dump(final_model, ROOT / "pcos_model.pkl")
        joblib.dump(transform, ROOT / "scaler.pkl")
        joblib.dump(list(X.columns), ROOT / "feature_names.pkl")
    report = {
        "dataset": DATA_PATH.name,
        "rows": int(len(data)),
        "features": int(X.shape[1]),
        "feature_names": list(X.columns),
        "target": TARGET,
        "target_counts": {str(int(label)): int(count) for label, count in y.value_counts().sort_index().items()},
        "split": {
            "test_size": 0.2,
            "stratified": True,
            "random_state": RANDOM_STATE,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
        "preprocessing": [
            "Drop the target, patient file identifier, and spreadsheet artifact columns.",
            "Convert feature columns to numeric values; unparseable values become missing.",
            "Impute missing values with medians fitted on training rows only.",
            "Standardize features using training rows only.",
        ],
        "models_compared": {
            name: {"estimator": type(estimator).__name__, "parameters": estimator.get_params(deep=False)}
            for name, estimator in candidates.items()
        },
        "cross_validation": {
            "folds": 5,
            "type": "StratifiedKFold",
            "shuffle": True,
            "random_state": RANDOM_STATE,
            "search_scoring": "accuracy",
            "grid_search_parameter_grid": {
                "n_estimators": [200, 400],
                "max_depth": [None, 8, 16],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt", 0.8],
            },
        },
        "best_extra_trees_parameters": search.best_params_,
        "best_cv_accuracy": float(search.best_score_),
        "software_versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "artifacts_saved": save_artifacts,
        "models": results,
    }
    report_json = json.dumps(report, indent=2)
    (ROOT / "model_results.json").write_text(report_json, encoding="utf-8")
    frontend_report = ROOT / "frontend" / "model_results.json"
    if frontend_report.parent.exists():
        frontend_report.write_text(report_json, encoding="utf-8")
    print(json.dumps({"best_parameters": search.best_params_, "results": results}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare PCOS classifiers and evaluate tuned Extra Trees.")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate evaluation reports without replacing the saved model artifacts.",
    )
    args = parser.parse_args()
    main(save_artifacts=not args.report_only)
