"""Gradient-boosted tree models: XGBoost and LightGBM.

Both are configured for **probability quality**, not for maximum AUC. The
differences from a stock classification setup are deliberate:

* ``objective`` / ``metric`` is log loss, which is a proper scoring rule —
  it is minimised by telling the truth. Optimising AUC instead would reward
  ranking and be indifferent to the scale of the output, which is the part
  Phase 5 depends on.
* **No class rebalancing.** See :mod:`backend.ml.models.base` for why.
* **Conservative depth and strong regularisation.** With tens of features and a
  ~11% base rate, deep trees memorise individual horses. Shallow trees plus
  ``min_child_weight`` keep leaves populated enough for their probabilities to
  mean something.
* **Early stopping on a chronologically later slice.** Stopping on a random
  sample of the training period would pick an iteration count that suits the
  past; stopping on later races picks one that survives the passage of time.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from backend.ml.models.base import RacingModel
from backend.utils.logging import get_logger

logger = get_logger(__name__, channel="model")

DEFAULT_EARLY_STOPPING_ROUNDS = 50


class XGBoostModel(RacingModel):
    """XGBoost gradient boosting."""

    name = "xgboost"
    needs_scaling = False

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "n_estimators": 600,
            "learning_rate": 0.05,
            "max_depth": 4,
            "min_child_weight": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 2.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_jobs": -1,
        }

    def _build_estimator(self) -> Any:
        from xgboost import XGBClassifier

        params = dict(self.params)
        params.setdefault("early_stopping_rounds", DEFAULT_EARLY_STOPPING_ROUNDS)
        return XGBClassifier(random_state=self.seed, **params)

    def _fit_estimator(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series,
        X_valid: pd.DataFrame | None,  # noqa: N803
        y_valid: pd.Series | None,
    ) -> None:
        if X_valid is not None and y_valid is not None and len(X_valid):
            self.estimator.fit(X, y, eval_set=[(X_valid, y_valid)], verbose=False)
            self.best_iteration = int(getattr(self.estimator, "best_iteration", 0) or 0)
        else:
            # No validation slice: early stopping is impossible, so disable it
            # rather than let XGBoost raise part-way through a training run.
            self.estimator.set_params(early_stopping_rounds=None)
            self.estimator.fit(X, y, verbose=False)
            logger.warning(
                "xgboost fitted without early stopping (no validation set supplied)",
                extra={"n_estimators": self.params.get("n_estimators")},
            )


class LightGBMModel(RacingModel):
    """LightGBM gradient boosting."""

    name = "lightgbm"
    needs_scaling = False

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "n_estimators": 600,
            "learning_rate": 0.05,
            # LightGBM grows leaf-wise, so depth is controlled by num_leaves.
            # 31 is the default and far too permissive for this base rate.
            "num_leaves": 15,
            "max_depth": 5,
            "min_child_samples": 40,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 2.0,
            "objective": "binary",
            "n_jobs": -1,
            "verbose": -1,
        }

    def _build_estimator(self) -> Any:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(random_state=self.seed, **self.params)

    def _fit_estimator(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series,
        X_valid: pd.DataFrame | None,  # noqa: N803
        y_valid: pd.Series | None,
    ) -> None:
        import lightgbm as lgb

        if X_valid is not None and y_valid is not None and len(X_valid):
            # LightGBM 4.7 renamed ``eval_set`` to ``eval_X``/``eval_y``. Both
            # spellings are supported across the version range we allow, so pick
            # whichever this install accepts rather than pinning a version.
            import inspect

            signature = inspect.signature(self.estimator.fit)
            validation = (
                {"eval_X": X_valid, "eval_y": y_valid}
                if "eval_X" in signature.parameters
                else {"eval_set": [(X_valid, y_valid)]}
            )
            self.estimator.fit(
                X,
                y,
                eval_metric="binary_logloss",
                callbacks=[
                    lgb.early_stopping(DEFAULT_EARLY_STOPPING_ROUNDS, verbose=False),
                    lgb.log_evaluation(0),
                ],
                **validation,
            )
            self.best_iteration = int(getattr(self.estimator, "best_iteration_", 0) or 0)
        else:
            self.estimator.fit(X, y, callbacks=[lgb.log_evaluation(0)])
            logger.warning("lightgbm fitted without early stopping (no validation set supplied)")


__all__ = ["DEFAULT_EARLY_STOPPING_ROUNDS", "LightGBMModel", "XGBoostModel"]
