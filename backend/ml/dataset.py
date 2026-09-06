"""Training dataset assembly and the train/validation/test contract.

Splits are by **date**, never at random. The reasoning is in
:mod:`backend.research.splits`; the short version is that a random split puts
runners from the same race on both sides and lets form features computed from
test-period races into training, which inflates every metric without looking
like a bug.

Four windows, not three
=======================

The specification asks for train / validation / test. This module produces four,
because the validation window has two jobs that must not be done with the same
rows:

``train``        fit the model
``valid_stop``   early stopping — the model is *chosen* using these rows
``valid_calib``  fit the probability calibrator
``test``         touched exactly once, at the end

If the calibrator were fitted on the same rows that chose the stopping point, it
would see predictions that are already optimistic on those rows and would learn
a correction that is too gentle. The two halves are split chronologically, so
``valid_calib`` is also the more recent — the better proxy for test conditions.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd
from sklearn.metrics import roc_auc_score
from sqlalchemy.orm import Session

from backend.features.base import EVENT_TIME, FeatureConfig
from backend.research.dataset import CATEGORICAL_COLUMNS, ResearchDatasetBuilder
from backend.research.splits import TemporalLeakageError
from backend.utils.logging import get_logger, safe_extra

logger = get_logger(__name__, channel="model")

TARGET = "won"

#: Windows from the Phase 4 specification. Real UK racing history is long enough
#: to support them; a short dataset will need :meth:`SplitConfig.adaptive`.
DEFAULT_TRAIN_START = date(2018, 1, 1)
DEFAULT_TRAIN_END = date(2023, 12, 31)
DEFAULT_VALID_START = date(2024, 1, 1)
DEFAULT_VALID_END = date(2024, 12, 31)
DEFAULT_TEST_START = date(2025, 1, 1)
DEFAULT_TEST_END = date(2026, 12, 31)

#: Days dropped after the training window. Rolling features span several weeks,
#: so a race immediately after the boundary shares most of its history with
#: training. The embargo costs a little data and buys an honest estimate.
DEFAULT_EMBARGO_DAYS = 14


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Date boundaries for the four windows."""

    train_start: date = DEFAULT_TRAIN_START
    train_end: date = DEFAULT_TRAIN_END
    valid_start: date = DEFAULT_VALID_START
    valid_end: date = DEFAULT_VALID_END
    test_start: date = DEFAULT_TEST_START
    test_end: date = DEFAULT_TEST_END
    embargo_days: int = DEFAULT_EMBARGO_DAYS
    #: Fraction of the validation window used for early stopping; the remainder
    #: (the more recent part) calibrates.
    calibration_fraction: float = 0.5

    def __post_init__(self) -> None:
        if not self.train_start < self.train_end < self.valid_start < self.valid_end < self.test_start:
            raise TemporalLeakageError(
                "split windows must be strictly ordered: "
                f"train {self.train_start}..{self.train_end}, "
                f"valid {self.valid_start}..{self.valid_end}, "
                f"test from {self.test_start}"
            )
        if not 0 < self.calibration_fraction < 1:
            raise ValueError("calibration_fraction must be in (0, 1)")

    @classmethod
    def adaptive(cls, frame: pd.DataFrame, *, column: str = "race_date", **overrides: Any) -> SplitConfig:
        """Derive proportional windows (60/20/20) from the data actually present.

        For use when the dataset does not span the specified calendar windows —
        a short backfill, or the synthetic season. It keeps the *ordering*
        guarantee, which is the part that matters, while accepting whatever
        history exists.
        """
        dates = pd.to_datetime(frame[column], errors="coerce").dt.date.dropna()
        if dates.empty:
            raise ValueError("cannot derive splits from a frame with no usable dates")

        ordered = sorted(dates.unique())
        train_cut = ordered[int(len(ordered) * 0.6)]
        valid_cut = ordered[int(len(ordered) * 0.8)]

        defaults: dict[str, Any] = {
            "train_start": ordered[0],
            "train_end": train_cut,
            "valid_start": _next_day(train_cut),
            "valid_end": valid_cut,
            "test_start": _next_day(valid_cut),
            "test_end": ordered[-1],
            "embargo_days": 0,
        }
        defaults.update(overrides)
        return cls(**defaults)

    def describe(self) -> dict[str, str]:
        return {
            "train": f"{self.train_start} → {self.train_end}",
            "valid": f"{self.valid_start} → {self.valid_end}",
            "test": f"{self.test_start} → {self.test_end}",
            "embargo_days": str(self.embargo_days),
        }


def _next_day(value: date) -> date:
    from datetime import timedelta

    return value + timedelta(days=1)


@dataclass(slots=True)
class TrainingData:
    """The four windows, plus everything needed to reproduce them."""

    train: pd.DataFrame
    valid_stop: pd.DataFrame
    valid_calib: pd.DataFrame
    test: pd.DataFrame

    numeric_features: list[str]
    categorical_features: list[str]
    target: str = TARGET
    config: SplitConfig = field(default_factory=SplitConfig)
    feature_version: str = "v1"

    # ------------------------------------------------------------------
    @property
    def valid(self) -> pd.DataFrame:
        """Both validation halves, for reporting."""
        return pd.concat([self.valid_stop, self.valid_calib], ignore_index=True)

    @property
    def feature_names(self) -> list[str]:
        return [*self.numeric_features, *self.categorical_features]

    def X(self, split: str) -> pd.DataFrame:  # noqa: N802 - conventional in ML code
        return self.frame(split)[self.feature_names]

    def y(self, split: str) -> pd.Series:
        return self.frame(split)[self.target]

    def frame(self, split: str) -> pd.DataFrame:
        try:
            return {
                "train": self.train,
                "valid": self.valid,
                "valid_stop": self.valid_stop,
                "valid_calib": self.valid_calib,
                "test": self.test,
            }[split]
        except KeyError:
            raise KeyError(f"unknown split {split!r}") from None

    def summary(self) -> dict[str, Any]:
        return {
            "splits": {
                name: {
                    "rows": len(frame),
                    "races": int(frame["race_id"].nunique()) if not frame.empty else 0,
                    "winners": int(frame[self.target].sum()) if not frame.empty else 0,
                    "from": str(frame["race_date"].min()) if not frame.empty else None,
                    "to": str(frame["race_date"].max()) if not frame.empty else None,
                }
                for name, frame in (
                    ("train", self.train),
                    ("valid_stop", self.valid_stop),
                    ("valid_calib", self.valid_calib),
                    ("test", self.test),
                )
            },
            "numeric_features": len(self.numeric_features),
            "categorical_features": len(self.categorical_features),
            "feature_version": self.feature_version,
        }

    def render(self) -> str:
        lines = ["Training data", "-------------"]
        for name, info in self.summary()["splits"].items():
            lines.append(
                f"  {name:<12} {info['rows']:>7} rows  {info['races']:>6} races  "
                f"{info['from']} → {info['to']}"
            )
        lines.append(
            f"  features     {len(self.numeric_features)} numeric + "
            f"{len(self.categorical_features)} categorical"
        )
        return "\n".join(lines)

    def assert_no_leakage(self) -> None:
        """Fail loudly if the windows overlap in time or share a race."""
        frames = [
            ("train", self.train),
            ("valid_stop", self.valid_stop),
            ("valid_calib", self.valid_calib),
            ("test", self.test),
        ]
        populated = [(name, frame) for name, frame in frames if not frame.empty]

        for (earlier_name, earlier), (later_name, later) in itertools.pairwise(populated):
            latest = pd.to_datetime(earlier["race_date"]).max()
            earliest = pd.to_datetime(later["race_date"]).min()
            if latest >= earliest:
                raise TemporalLeakageError(
                    f"{earlier_name} reaches {latest.date()} but {later_name} starts {earliest.date()}"
                )

        seen: dict[str, str] = {}
        for name, frame in populated:
            for race_id in frame["race_id"].unique():
                if race_id in seen:
                    raise TemporalLeakageError(f"race {race_id} appears in both {seen[race_id]} and {name}")
                seen[race_id] = name


class TrainingDatasetBuilder:
    """Builds :class:`TrainingData` from the warehouse."""

    def __init__(
        self,
        session: Session,
        config: FeatureConfig | None = None,
        *,
        apply_data_quality: bool = True,
    ) -> None:
        self.session = session
        self.builder = ResearchDatasetBuilder(session, config, apply_data_quality=apply_data_quality)

    def build(
        self,
        split: SplitConfig | None = None,
        *,
        min_runs: int = 0,
        require_market: bool = False,
        adaptive_if_empty: bool = False,
        feature_version: str = "v1",
        drop_feature_prefixes: tuple[str, ...] = (),
    ) -> TrainingData:
        """Assemble the dataset and cut it into the four windows.

        ``adaptive_if_empty`` falls back to proportional windows when the
        configured calendar windows produce no training rows. It exists so the
        pipeline is demonstrable on a short history; it must never be used to
        silently reinterpret a production configuration, so it logs loudly.
        """
        split = split or SplitConfig()

        dataset = self.builder.build(
            date_from=split.train_start,
            date_to=split.test_end,
            min_runs=min_runs,
            require_market=require_market,
            include_categoricals=True,
        )
        if dataset.rows == 0:
            raise ValueError("no rows available to build a training dataset")

        frame = dataset.frame
        numeric = [name for name in dataset.feature_names if name not in CATEGORICAL_COLUMNS]
        if drop_feature_prefixes:
            before = len(numeric)
            numeric = [name for name in numeric if not name.startswith(tuple(drop_feature_prefixes))]
            logger.info(
                "feature families excluded",
                extra=safe_extra({"prefixes": list(drop_feature_prefixes), "dropped": before - len(numeric)}),
            )
        categorical = [name for name in CATEGORICAL_COLUMNS if name in frame.columns]

        windows = self._cut(frame, split)
        if adaptive_if_empty and windows["train"].empty:
            logger.warning(
                "configured split windows produced no training rows; falling back to "
                "proportional windows derived from the data",
                extra=safe_extra(split.describe()),
            )
            split = SplitConfig.adaptive(frame)
            windows = self._cut(frame, split)

        data = TrainingData(
            train=windows["train"],
            valid_stop=windows["valid_stop"],
            valid_calib=windows["valid_calib"],
            test=windows["test"],
            numeric_features=numeric,
            categorical_features=categorical,
            config=split,
            feature_version=feature_version,
        )
        data.assert_no_leakage()
        logger.info("training dataset built", extra=safe_extra(data.summary()["splits"]["train"]))
        return data

    # ------------------------------------------------------------------
    @staticmethod
    def _cut(frame: pd.DataFrame, split: SplitConfig) -> dict[str, pd.DataFrame]:
        from datetime import timedelta

        dates = pd.to_datetime(frame["race_date"], errors="coerce").dt.date
        embargo_end = split.train_end + timedelta(days=split.embargo_days)

        train = frame[(dates >= split.train_start) & (dates <= split.train_end)]
        # The embargo carves a gap out of the *validation* side of the boundary,
        # so no validation race shares rolling history with a training race.
        valid = frame[(dates > embargo_end) & (dates <= split.valid_end) & (dates >= split.valid_start)]
        test = frame[(dates >= split.test_start) & (dates <= split.test_end)]

        valid = valid.sort_values("race_date", kind="stable")
        stop, calib = TrainingDatasetBuilder._halve_by_date(valid, split.calibration_fraction)

        return {
            "train": train.reset_index(drop=True),
            "valid_stop": stop.reset_index(drop=True),
            "valid_calib": calib.reset_index(drop=True),
            "test": test.reset_index(drop=True),
        }

    @staticmethod
    def _halve_by_date(frame: pd.DataFrame, fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the validation window chronologically on a *date* boundary.

        A race-count boundary looks equivalent and is not: several races share a
        date, so cutting on the nth race routinely puts two races from the same
        afternoon on opposite sides. That leaves the two halves overlapping in
        time, which is precisely what every other split in this system forbids.
        Cutting on the date keeps the ordering strict.
        """
        if frame.empty:
            return frame, frame

        dates = pd.to_datetime(frame["race_date"], errors="coerce").dt.date
        unique_dates = sorted(dates.dropna().unique())
        if len(unique_dates) < 2:
            return frame, frame.iloc[0:0]

        cut_index = min(len(unique_dates) - 1, max(1, int(len(unique_dates) * fraction)))
        cut_date = unique_dates[cut_index - 1]

        mask = dates <= cut_date
        return frame[mask], frame[~mask]


#: A single feature that separates winners from losers this perfectly is not a
#: feature, it is the answer. Genuine racing signals top out far below this: even
#: the bookmakers' own price only reaches ~0.75 AUC on its own.
LEAKAGE_AUC_THRESHOLD = 0.999


@dataclass(frozen=True, slots=True)
class LeakSuspect:
    """A feature that predicts the target implausibly well."""

    feature: str
    auc: float

    def __str__(self) -> str:
        return f"{self.feature} (AUC {self.auc:.4f})"


def detect_leaky_features(
    frame: pd.DataFrame,
    feature_names: list[str],
    target: str = TARGET,
    *,
    threshold: float = LEAKAGE_AUC_THRESHOLD,
    sample: int = 20_000,
) -> list[LeakSuspect]:
    """Find single features that alone separate winners from losers.

    A name-based denylist only catches leaks somebody already thought of. This
    catches them by behaviour: any column achieving near-perfect AUC on its own
    is the target wearing a different name, whatever that name happens to be.

    It exists because exactly that happened here. ``load_labels`` returned an
    ``is_winner`` boolean, the outcome denylist listed ``won`` but not
    ``is_winner``, and a bool counts as numeric — so the target sailed into the
    feature matrix and every model scored a perfect 1.000 AUC.
    """
    if frame.empty or target not in frame.columns:
        return []

    subset = frame.sample(sample, random_state=0) if len(frame) > sample else frame
    truth = pd.to_numeric(subset[target], errors="coerce").fillna(0).astype(int).to_numpy()
    if truth.sum() in (0, truth.size):
        return []

    suspects: list[LeakSuspect] = []
    for name in feature_names:
        if name not in subset.columns:
            continue
        values = pd.to_numeric(subset[name], errors="coerce")
        if values.notna().sum() < 2 or values.nunique(dropna=True) < 2:
            continue
        filled = values.fillna(values.median()).to_numpy()
        try:
            auc = float(roc_auc_score(truth, filled))
        except ValueError:  # pragma: no cover - constant column after fill
            continue
        # Perfect *negative* separation is just as damning as perfect positive.
        strength = max(auc, 1.0 - auc)
        if strength >= threshold:
            suspects.append(LeakSuspect(feature=name, auc=strength))

    return sorted(suspects, key=lambda suspect: suspect.auc, reverse=True)


__all__ = [
    "DEFAULT_EMBARGO_DAYS",
    "EVENT_TIME",
    "LEAKAGE_AUC_THRESHOLD",
    "TARGET",
    "LeakSuspect",
    "SplitConfig",
    "TrainingData",
    "TrainingDatasetBuilder",
    "detect_leaky_features",
]
