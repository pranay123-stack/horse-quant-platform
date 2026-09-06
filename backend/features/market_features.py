"""Market features: the bookmakers' own opinion, and how it moved.

The market is the benchmark. Bookmakers' prices are the most accurate publicly
available probability estimate for a horse race, and a model that cannot beat
them after the overround has no edge. Every market feature here exists so the
model can be measured against that benchmark — and, where the market is slow, so
it can learn where the market is wrong.

**Odds movement** is the highest-value feature in this module. A horse whose
price shortens from 10.0 to 6.0 has attracted informed money; drift in the other
direction often reflects a stable that has gone quiet on it. The movement is
frequently more informative than the price level itself.

Leakage controls
----------------
* Only quotes recorded **at or before the off** are used. A price timestamped
  after the off is not market opinion, it is the result.
* Quotes on races with **no off-time** are discarded entirely: we cannot prove
  they predate the start, and an unprovable quote is treated as unusable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MARKET_FEATURE_COLUMNS: tuple[str, ...] = (
    "mkt_open_odds",
    "mkt_latest_odds",
    "mkt_mean_odds",
    "mkt_odds_change_pct",
    "mkt_shortened",
    "mkt_implied_prob",
    "mkt_implied_prob_norm",
    "mkt_overround",
    "mkt_rank",
    "mkt_is_favourite",
    "mkt_bookmaker_count",
    "mkt_quote_count",
    "mkt_has_market",
)


def _empty_market_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["race_id", "horse_id", *MARKET_FEATURE_COLUMNS])


def build_market_state(odds: pd.DataFrame) -> pd.DataFrame:
    """Collapse the quote history into one pre-race market view per runner."""
    if odds.empty:
        return _empty_market_frame()

    frame = odds.copy()
    frame["decimal_odds"] = pd.to_numeric(frame["decimal_odds"], errors="coerce")
    frame["recorded_at"] = pd.to_datetime(frame["recorded_at"], utc=True, errors="coerce")
    frame["off_time"] = pd.to_datetime(frame["off_time"], utc=True, errors="coerce")

    # Drop anything we cannot prove is pre-race, and anything unpriced.
    frame = frame[frame["decimal_odds"].notna() & (frame["decimal_odds"] > 1)]
    frame = frame[frame["recorded_at"].notna() & frame["off_time"].notna()]
    frame = frame[frame["recorded_at"] <= frame["off_time"]]
    if frame.empty:
        return _empty_market_frame()

    frame = frame.sort_values(["race_id", "horse_id", "bookmaker", "recorded_at"], kind="stable")
    runner_keys = ["race_id", "horse_id"]
    book_keys = [*runner_keys, "bookmaker"]

    # Per bookmaker: its first and last pre-race price for this runner.
    first_per_book = frame.groupby(book_keys, sort=False, observed=True).first().reset_index()
    last_per_book = frame.groupby(book_keys, sort=False, observed=True).last().reset_index()

    # Across bookmakers: the best price obtainable, which is what a bettor takes.
    opening = (
        first_per_book.groupby(runner_keys, sort=False, observed=True)["decimal_odds"]
        .max()
        .rename("mkt_open_odds")
    )
    latest = last_per_book.groupby(runner_keys, sort=False, observed=True).agg(
        mkt_latest_odds=("decimal_odds", "max"),
        mkt_mean_odds=("decimal_odds", "mean"),
        mkt_bookmaker_count=("bookmaker", "nunique"),
    )
    quote_counts = frame.groupby(runner_keys, sort=False, observed=True).size().rename("mkt_quote_count")

    market = pd.concat([latest, opening, quote_counts], axis=1).reset_index()

    # Drift: positive means the price lengthened (support ebbing away).
    market["mkt_odds_change_pct"] = (
        (market["mkt_latest_odds"] - market["mkt_open_odds"]) / market["mkt_open_odds"]
    ).replace([np.inf, -np.inf], np.nan)
    market["mkt_shortened"] = (market["mkt_odds_change_pct"] < 0).astype(int)

    market["mkt_implied_prob"] = 1.0 / market["mkt_latest_odds"]

    # Overround: the sum of implied probabilities across the field. Dividing by
    # it removes the bookmaker's margin and yields probabilities that sum to 1 --
    # the fair benchmark the model must beat.
    overround = market.groupby("race_id", sort=False, observed=True)["mkt_implied_prob"].transform("sum")
    market["mkt_overround"] = overround
    market["mkt_implied_prob_norm"] = market["mkt_implied_prob"] / overround.where(overround > 0)

    market["mkt_rank"] = (
        market.groupby("race_id", sort=False, observed=True)["mkt_implied_prob"]
        .rank(ascending=False, method="min")
        .astype("float64")
    )
    market["mkt_is_favourite"] = (market["mkt_rank"] == 1).astype(int)
    market["mkt_has_market"] = 1

    return market[["race_id", "horse_id", *MARKET_FEATURE_COLUMNS]]


def build_market_features(targets: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Attach market features, leaving them null when no market was captured.

    Missing market data is imputed as *absent*, never as a neutral price. A
    fabricated price would look to the strategy engine like a real betting
    opportunity, which is the most expensive kind of missing-data bug in this
    system.
    """
    market = build_market_state(odds)
    if targets.empty:
        for column in MARKET_FEATURE_COLUMNS:
            targets[column] = pd.Series(dtype="float64")
        return targets

    frame = targets.merge(market, on=["race_id", "horse_id"], how="left")
    # Coerce before filling: a left join onto an empty market frame leaves these
    # as object dtype, and ``fillna`` on object columns is deprecated.
    for column in ("mkt_has_market", "mkt_is_favourite"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int)
    for column in ("mkt_quote_count", "mkt_bookmaker_count"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    return frame


__all__ = ["MARKET_FEATURE_COLUMNS", "build_market_features", "build_market_state"]
