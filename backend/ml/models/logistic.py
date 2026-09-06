"""Logistic regression baseline.

The point of a baseline is not to win. It is to answer "is the complicated model
earning its complexity?" — and to be interpretable enough that a wrong sign in a
coefficient exposes a broken feature.

L2 regularisation is on by default. Many of these features are strongly
correlated by construction (``horse_form_last3_avg_pos`` and
``horse_form_last5_avg_pos`` share four of five runs), and unregularised logistic
regression responds to collinearity with huge cancelling coefficients that
destroy both stability and interpretability.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression

from backend.ml.models.base import RacingModel


class LogisticRegressionModel(RacingModel):
    """Regularised logistic regression on standardised features."""

    name = "logistic_regression"
    #: Unlike trees, this genuinely needs standardised inputs — both for
    #: convergence and to make coefficients comparable to one another.
    needs_scaling = True

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        # L2 is the default; naming it explicitly is deprecated from
        # scikit-learn 1.8, and ``n_jobs`` has been a no-op for lbfgs since the
        # same release. Regularisation strength is set through ``C`` alone.
        return {
            "C": 1.0,
            # lbfgs handles the dense, standardised matrix well and needs no
            # extra configuration for the binary case.
            "solver": "lbfgs",
            "max_iter": 2000,
        }

    def _build_estimator(self) -> LogisticRegression:
        return LogisticRegression(random_state=self.seed, **self.params)

    def coefficients(self, *, top: int | None = None) -> pd.DataFrame:
        """Signed coefficients — the readable part of the baseline.

        On standardised inputs these are directly comparable: a coefficient of
        +0.4 moves the log-odds of winning by 0.4 for a one-standard-deviation
        increase in that feature.
        """
        frame = self.feature_importance()
        if frame.empty:
            return frame
        raw = self._raw_importance()
        if raw is not None:
            lookup = dict(zip(self.preprocessor.output_names, raw, strict=True))
            frame["coefficient"] = frame["feature"].map(lookup)
            frame["direction"] = frame["coefficient"].apply(lambda value: "+" if value >= 0 else "-")
        return frame.head(top) if top else frame


__all__ = ["LogisticRegressionModel"]
