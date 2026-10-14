"""
Foundation of Data Science – Assessment 3, Objective 2
Member 3: modelling, validation, optimisation and integration.

Expected input files in the same folder:
  HIT140_Final_dataset_Assessment-3_group-20.csv
      104 match-level rows; target goal_diff
  HIT140_Regression_2_2_208_team_match_dataset_7507.csv
      208 team-match rows; target goals_scored

The predictor lists and file/target names below are configured for these supplied datasets.
Train/test splitting and cross-validation are grouped by match_id so paired team rows from
the same match do not leak across folds in Task 2.2.

Install: pip install pandas numpy scikit-learn matplotlib
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split, KFold, GroupKFold, GroupShuffleSplit,
    GridSearchCV, cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_SPLITS = 5
OUT = Path("member3_outputs")
OUT.mkdir(exist_ok=True)

PREDICTORS_21 = [
    "rank_diff", "log_value_diff", "age_diff", "host_diff",
    "is_knockout", "prior_ppg_diff", "prior_gd_diff", "rest_days_diff"
]
PREDICTORS_22 = [
    "rank_diff", "log_value_diff", "prior_ppg_diff", "rest_days_diff",
    "team_recent_goals_avg", "opponent_recent_goals_conceded_avg",
    "team_recent_goals_conceded_avg", "opponent_recent_goals_avg"
]

TASKS = [
    {"name": "LR_2_1_goal_difference", "file": "HIT140_Final_dataset_Assessment-3_group-20.csv",
     "target": "goal_diff", "predictors": PREDICTORS_21,
     "expected_rows": 104, "group_column": "match_id"},
    {"name": "LR_2_2_goals_scored", "file": "HIT140_Regression_2_2_208_team_match_dataset_7507.csv",
     "target": "goals_scored", "predictors": PREDICTORS_22,
     "expected_rows": 208, "group_column": "match_id"},
]

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def validate_dataset(df, task):
    """Validate the required raw row count, predictors, target and usable target values."""
    if len(df) != task["expected_rows"]:
        raise ValueError(
            f'{task["name"]}: brief requires exactly {task["expected_rows"]} rows, '
            f'but {len(df)} were found. Resolve the dataset before modelling.'
        )
    predictors = task["predictors"]
    if len(predictors) != 8 or len(set(predictors)) != 8:
        raise ValueError(f'{task["name"]}: specify exactly 8 unique explanatory variables.')
    if task["target"] in predictors:
        raise ValueError("The target variable cannot also be a predictor.")
    required = predictors + [task["target"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'{task["name"]}: missing required columns: {missing}')
    data = df[required].copy()
    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    if data[task["target"]].isna().any():
        n = int(data[task["target"]].isna().sum())
        raise ValueError(
            f'{task["name"]}: target has {n} missing/non-numeric values. '
            'Resolve these from verified sources; do not impute the outcome.'
        )
    if data[task["target"]].nunique() < 2:
        raise ValueError(f'{task["name"]}: target has no variation.')
    # Predictors may have missing values; imputation is fitted inside each training fold.
    groups = None
    group_col = task.get("group_column")
    if group_col and group_col in df.columns:
        groups = df[group_col].copy()
        if groups.isna().any():
            raise ValueError(f'{task["name"]}: {group_col} contains missing IDs.')
    elif task["name"] == "LR_2_2_goals_scored":
        warnings.warn(
            "match_id column not found. Team rows from the same match may cross folds; "
            "add a match_id identifier to enable grouped validation."
        )
    return data, groups

def make_cv(groups_train):
    if groups_train is not None:
        n_groups = pd.Series(groups_train).nunique()
        if n_groups < CV_SPLITS:
            raise ValueError(f"Need at least {CV_SPLITS} unique groups for grouped CV.")
        return GroupKFold(n_splits=CV_SPLITS)
    return KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

def make_pipeline(model):
    # Imputation and scaling are learned inside each fold, preventing preprocessing leakage.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", model),
    ])

def run_task(task):
    path = Path(task["file"])
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
    raw = pd.read_csv(path)
    data, groups = validate_dataset(raw, task)
    X = data[task["predictors"]]
    y = data[task["target"]]

    # Group by match whenever match_id exists. In Task 2.2, this prevents the two
    # team-level records for one match appearing on both sides of the split.
    if groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        groups_train = None

    cv = make_cv(groups_train)
    common = {"cv": cv, "scoring": "neg_root_mean_squared_error", "n_jobs": -1}
    candidates = {
        "Mean baseline": make_pipeline(DummyRegressor(strategy="mean")),
        "Ordinary least squares": make_pipeline(LinearRegression()),
        "Ridge (tuned)": GridSearchCV(
            make_pipeline(Ridge()),
            {"model__alpha": [0.01, 0.1, 1, 10, 100]}, **common
        ),
        "Lasso (tuned)": GridSearchCV(
            make_pipeline(Lasso(max_iter=30000)),
            {"model__alpha": [0.001, 0.01, 0.1, 1, 10]}, **common
        ),
    }

    records, fitted, cv_rmse = [], {}, {}
    for name, estimator in candidates.items():
        if isinstance(estimator, GridSearchCV):
            if groups_train is not None:
                estimator.fit(X_train, y_train, groups=groups_train)
            else:
                estimator.fit(X_train, y_train)
            score = -float(estimator.best_score_)
            params = str(estimator.best_params_)
        else:
            estimator.fit(X_train, y_train)
            scores = cross_val_score(
                estimator, X_train, y_train, groups=groups_train, cv=cv,
                scoring="neg_root_mean_squared_error", n_jobs=-1
            )
            score = -float(scores.mean())
            params = "default"
        fitted[name] = estimator
        cv_rmse[name] = score
        pred = estimator.predict(X_test)
        records.append({
            "task": task["name"], "model": name,
            "training_CV_RMSE": score,
            "test_MAE": mean_absolute_error(y_test, pred),
            "test_RMSE": rmse(y_test, pred),
            "test_R2": r2_score(y_test, pred),
            "best_parameters": params,
        })

    # Select using training CV only. Test metrics are reported for comparison but do not
    # determine the selected model.
    best_name = min(cv_rmse, key=cv_rmse.get)
    best_model = fitted[best_name]
    y_pred = best_model.predict(X_test)
    residuals = y_test - y_pred
    results = pd.DataFrame(records).sort_values("training_CV_RMSE")
    results.to_csv(OUT / f'{task["name"]}_model_comparison.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(y_test, y_pred, alpha=0.8)
    low = min(float(y_test.min()), float(np.min(y_pred)))
    high = max(float(y_test.max()), float(np.max(y_pred)))
    axes[0].plot([low, high], [low, high], linestyle="--")
    axes[0].set(title=f'{task["name"]}: observed vs predicted',
                xlabel="Observed", ylabel="Predicted")
    axes[1].scatter(y_pred, residuals, alpha=0.8)
    axes[1].axhline(0, linestyle="--")
    axes[1].set(title="Residuals vs predicted", xlabel="Predicted", ylabel="Residual")
    fig.tight_layout()
    fig.savefig(OUT / f'{task["name"]}_diagnostics.png', dpi=180)
    plt.close(fig)

    pi = permutation_importance(
        best_model, X_test, y_test, n_repeats=30,
        random_state=RANDOM_STATE, scoring="r2", n_jobs=-1
    )
    importance = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean_R2_decrease": pi.importances_mean,
        "importance_sd": pi.importances_std
    }).sort_values("importance_mean_R2_decrease", ascending=False)
    importance.to_csv(OUT / f'{task["name"]}_permutation_importance.csv', index=False)

    print(f"\\n=== {task['name']} ===")
    print(f"Raw rows validated: {len(raw)}; rows used: {len(data)}")
    print(f"Validation: {'grouped by match' if groups is not None else 'random row split'}")
    print(results.to_string(index=False))
    print(f"Selected by training CV: {best_name}")
    print("Selected-model test metrics:",
          {"MAE": round(mean_absolute_error(y_test, y_pred), 4),
           "RMSE": round(rmse(y_test, y_pred), 4),
           "R2": round(r2_score(y_test, y_pred), 4)})
    print(f"Outputs saved to: {OUT.resolve()}")
    return results

def main():
    shared = set(PREDICTORS_21).intersection(PREDICTORS_22)
    if len(shared) > 4:
        raise ValueError(
            f"The brief permits at most 4 shared explanatory variables; found {len(shared)}: {sorted(shared)}"
        )
    results = [run_task(task) for task in TASKS]
    pd.concat(results, ignore_index=True).to_csv(
        OUT / "combined_model_comparison.csv", index=False
    )
    print("Combined comparison saved to", (OUT / "combined_model_comparison.csv").resolve())

if __name__ == "__main__":
    main()
