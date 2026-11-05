"""Walk-forward LightGBM training producing out-of-sample predictions."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import lightgbm as lgb
import pandas as pd

from .cv import PurgedWalkForwardCV

log = logging.getLogger(__name__)

DEFAULT_PARAMS = dict(
    objective="regression",
    learning_rate=0.03,
    n_estimators=300,
    num_leaves=16,
    min_child_samples=200,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
)


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame        # columns: y_true, y_pred, fold
    feature_importance: pd.DataFrame  # rows: features, columns: folds (gain)


def walk_forward_predict(
    X: pd.DataFrame,
    y: pd.Series,
    cv: PurgedWalkForwardCV,
    params: dict | None = None,
) -> WalkForwardResult:
    """Refit the model on each training window and predict the next test window."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    preds, importance = [], {}
    n_splits = cv.get_n_splits(X)
    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
        model = lgb.LGBMRegressor(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_pred = pd.DataFrame(
            {"y_true": y.iloc[test_idx], "y_pred": model.predict(X.iloc[test_idx]), "fold": fold},
            index=X.index[test_idx],
        )
        preds.append(fold_pred)
        importance[fold] = pd.Series(
            model.booster_.feature_importance("gain"), index=X.columns
        )
        dates = fold_pred.index.get_level_values("date")
        log.info("fold %d/%d: test %s -> %s", fold + 1, n_splits, dates.min().date(), dates.max().date())
    return WalkForwardResult(
        predictions=pd.concat(preds).sort_index(level=["date", "symbol"]),
        feature_importance=pd.DataFrame(importance),
    )
