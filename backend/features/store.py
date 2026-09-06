"""Persisting and reloading feature vectors."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.features.feature_pipeline import feature_columns
from backend.features.scoring import SCORE_COLUMNS
from backend.models.features import FEATURE_VERSION, RaceFeatures
from backend.utils.logging import get_logger, safe_extra

logger = get_logger(__name__, channel="model")


def _clean(value: Any) -> Any:
    """JSON-safe scalar. ``NaN``/``inf`` become ``None``.

    ``json.dumps`` will happily emit a bare ``NaN`` token, which is invalid JSON
    and fails on read-back from PostgreSQL. Normalising here keeps the stored
    vector loadable.
    """
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, float | int) and not isinstance(value, bool):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    if hasattr(value, "item"):
        try:
            return _clean(value.item())
        except (ValueError, AttributeError):  # pragma: no cover - exotic dtypes
            return None
    if isinstance(value, pd.Timestamp | date):
        return value.isoformat()
    return value


def save_features(
    session: Session,
    frame: pd.DataFrame,
    *,
    feature_version: str = FEATURE_VERSION,
    batch_size: int = 1000,
) -> int:
    """Upsert feature rows. Returns how many were written.

    Idempotent on ``(race_id, horse_id, feature_version)``: re-running a build
    updates in place rather than duplicating, so a corrected pipeline can simply
    be re-run.
    """
    if frame.empty:
        return 0

    columns = feature_columns(frame)
    written = 0

    for start in range(0, len(frame), batch_size):
        chunk = frame.iloc[start : start + batch_size]
        keys = [(row.race_id, row.horse_id) for row in chunk.itertuples(index=False)]
        existing = {
            (row.race_id, row.horse_id): row
            for row in session.execute(
                select(RaceFeatures).where(
                    RaceFeatures.feature_version == feature_version,
                    RaceFeatures.race_id.in_([key[0] for key in keys]),
                )
            ).scalars()
        }

        for record in chunk.to_dict(orient="records"):
            key = (record["race_id"], record["horse_id"])
            vector = {column: _clean(record.get(column)) for column in columns}
            values: dict[str, Any] = {
                "race_date": record.get("race_date"),
                "horse_runs_before": _to_int(record.get("horse_runs_before")),
                "features": vector,
            }
            for score in SCORE_COLUMNS:
                values[score] = _clean(record.get(score))

            row = existing.get(key)
            if row is None:
                session.add(
                    RaceFeatures(
                        race_id=key[0],
                        horse_id=key[1],
                        feature_version=feature_version,
                        **values,
                    )
                )
            else:
                for name, value in values.items():
                    setattr(row, name, value)
            written += 1

        session.flush()

    session.commit()
    logger.info(
        "features stored",
        extra=safe_extra({"rows": written, "version": feature_version, "columns": len(columns)}),
    )
    return written


def _to_int(value: Any) -> int | None:
    cleaned = _clean(value)
    return int(cleaned) if cleaned is not None else None


def load_features(
    session: Session,
    *,
    date_from: date | None = None,
    date_to: date | None = None,
    race_ids: Sequence[str] | None = None,
    feature_version: str = FEATURE_VERSION,
) -> pd.DataFrame:
    """Reload stored feature vectors as a flat frame."""
    statement = select(RaceFeatures).where(RaceFeatures.feature_version == feature_version)
    if race_ids is not None:
        statement = statement.where(RaceFeatures.race_id.in_(race_ids))
    if date_from is not None:
        statement = statement.where(RaceFeatures.race_date >= date_from)
    if date_to is not None:
        statement = statement.where(RaceFeatures.race_date <= date_to)

    rows = session.execute(statement.order_by(RaceFeatures.race_date, RaceFeatures.race_id)).scalars().all()
    if not rows:
        return pd.DataFrame(columns=["race_id", "horse_id", "race_date", *SCORE_COLUMNS])

    records = []
    for row in rows:
        record: dict[str, Any] = {
            "race_id": row.race_id,
            "horse_id": row.horse_id,
            "race_date": row.race_date,
            "feature_version": row.feature_version,
        }
        record.update({score: getattr(row, score) for score in SCORE_COLUMNS})
        record.update(row.features or {})
        records.append(record)
    return pd.DataFrame(records)


__all__ = ["load_features", "save_features"]
