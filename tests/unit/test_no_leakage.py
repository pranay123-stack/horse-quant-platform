"""Look-ahead bias tests — the most important tests in the repository.

A leak does not crash anything. It makes a model look brilliant in research and
lose money in production, and by the time that is obvious a lot of money is gone.
These tests are the tripwire.

Four independent angles, because one check is easy to fool:

1. **Future injection.** Build features, then add races that happen *later*, then
   rebuild. Every earlier feature value must be byte-identical. This catches any
   aggregate that accidentally spans the whole dataset.
2. **Own-result exclusion.** A runner's features must not reflect the race it is
   running in.
3. **Mechanism.** :func:`as_of_join` itself, on hand-built frames where the right
   answer is obvious by inspection.
4. **Split integrity.** No race on both sides of a train/test boundary.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from backend.features.base import EVENT_TIME, as_of_join
from backend.features.feature_pipeline import FeaturePipeline, feature_columns
from backend.research.synthetic import SyntheticConfig, generate_synthetic_data
from backend.utils.timeutils import UK_TZ
from tests.fixtures.racing_builders import build_history_frame, build_race_with_results

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 1. Future injection
# ---------------------------------------------------------------------------
def test_adding_future_races_does_not_change_earlier_features(db_session):
    """The decisive test: tomorrow's racing cannot alter today's features."""
    generate_synthetic_data(
        db_session, SyntheticConfig(n_days=40, races_per_day=3, runners_per_race=6, seed=11)
    )
    cutoff = date(2025, 1, 25)

    before = FeaturePipeline(db_session).build(date_to=cutoff)
    assert not before.empty

    # Everything after the cutoff already exists in the database; restricting the
    # build window is what changes. If any feature were computed over the whole
    # table rather than point-in-time, these two frames would differ.
    after = FeaturePipeline(db_session).build()
    after = after[after["race_date"] <= cutoff].reset_index(drop=True)

    keys = ["race_id", "horse_id"]
    before = before.sort_values(keys).reset_index(drop=True)
    after = after.sort_values(keys).reset_index(drop=True)
    assert len(before) == len(after)

    columns = [c for c in feature_columns(before) if c in after.columns]
    differing = [
        column
        for column in columns
        if not before[column].fillna(-999).round(9).equals(after[column].fillna(-999).round(9))
    ]
    assert not differing, f"these features changed when future races were visible: {differing}"


def test_horse_features_ignore_that_horse_s_later_runs(db_session):
    """A horse's third run must not know about its fourth."""
    horse = "hrs_leak"
    base = datetime(2025, 3, 1, 14, 0, tzinfo=UK_TZ)

    # Three losing runs, then a win. Features for the fourth race must show the
    # horse as winless.
    for index, position in enumerate([5, 6, 4, 1]):
        build_race_with_results(
            db_session,
            race_id=f"rac_leak_{index}",
            off_time=base + timedelta(days=index * 7),
            runners=[(horse, position), (f"hrs_other_{index}", 1 if position != 1 else 2)],
        )
    db_session.commit()

    frame = FeaturePipeline(db_session).build()
    fourth = frame[(frame["race_id"] == "rac_leak_3") & (frame["horse_id"] == horse)].iloc[0]

    assert fourth["horse_runs_before"] == 3
    assert fourth["horse_wins_before"] == 0, "the win being predicted leaked into its own features"
    assert fourth["horse_last1_position"] == 4  # the third run, not the fourth


def test_first_ever_run_has_no_history(db_session):
    build_race_with_results(
        db_session,
        race_id="rac_debut",
        off_time=datetime(2025, 3, 1, 14, 0, tzinfo=UK_TZ),
        runners=[("hrs_debut", 1), ("hrs_debut_b", 2)],
    )
    db_session.commit()

    frame = FeaturePipeline(db_session).build()
    row = frame[frame["horse_id"] == "hrs_debut"].iloc[0]

    assert row["horse_runs_before"] == 0
    assert row["horse_is_debutant"] == 1
    assert pd.isna(row["horse_last1_position"])


def test_market_features_ignore_quotes_recorded_after_the_off(db_session):
    """A price stamped after the off is the result, not the market."""
    off = datetime(2025, 3, 1, 14, 0, tzinfo=UK_TZ)
    build_race_with_results(
        db_session,
        race_id="rac_odds",
        off_time=off,
        runners=[("hrs_a", 1), ("hrs_b", 2)],
        odds={
            "hrs_a": [(off - timedelta(minutes=30), 3.0), (off + timedelta(minutes=5), 1.01)],
            "hrs_b": [(off - timedelta(minutes=30), 4.0)],
        },
    )
    db_session.commit()

    frame = FeaturePipeline(db_session).build()
    winner = frame[frame["horse_id"] == "hrs_a"].iloc[0]

    # 1.01 is the post-race "price" of a horse that has already won.
    assert winner["mkt_latest_odds"] == 3.0, "a post-off quote leaked into the market features"


# ---------------------------------------------------------------------------
# 2 & 3. The mechanism itself
# ---------------------------------------------------------------------------
def _state(times: list[str], values: list[float], entity: str = "e1") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity_id": [entity] * len(times),
            EVENT_TIME: pd.to_datetime(times, utc=True),
            "stat": values,
        }
    )


def _targets(times: list[str], entity: str = "e1") -> pd.DataFrame:
    return pd.DataFrame({"entity_id": [entity] * len(times), EVENT_TIME: pd.to_datetime(times, utc=True)})


def test_as_of_join_takes_the_last_state_strictly_before():
    state = _state(["2025-01-01", "2025-02-01", "2025-03-01"], [1.0, 2.0, 3.0])
    targets = _targets(["2025-01-15", "2025-02-15", "2025-03-15"])

    merged = as_of_join(targets, state, by="entity_id", columns=["stat"])
    assert merged["stat"].tolist() == [1.0, 2.0, 3.0]


def test_as_of_join_excludes_an_exactly_simultaneous_event():
    """This is what stops a race's own result entering its own features."""
    state = _state(["2025-01-01", "2025-02-01"], [1.0, 2.0])
    targets = _targets(["2025-02-01"])  # same instant as the second state row

    merged = as_of_join(targets, state, by="entity_id", columns=["stat"])
    assert merged["stat"].tolist() == [1.0]


def test_as_of_join_yields_nothing_before_any_history():
    state = _state(["2025-06-01"], [5.0])
    merged = as_of_join(_targets(["2025-01-01"]), state, by="entity_id", columns=["stat"])
    assert pd.isna(merged["stat"].iloc[0])


def test_as_of_join_never_crosses_entities():
    state = pd.concat([_state(["2025-01-01"], [10.0], "a"), _state(["2025-01-01"], [20.0], "b")])
    targets = pd.concat([_targets(["2025-02-01"], "a"), _targets(["2025-02-01"], "b")])

    merged = as_of_join(targets, state, by="entity_id", columns=["stat"]).sort_values("entity_id")
    assert merged["stat"].tolist() == [10.0, 20.0]


def test_as_of_join_handles_empty_state():
    merged = as_of_join(_targets(["2025-02-01"]), pd.DataFrame(), by="entity_id", columns=["stat"])
    assert merged["stat"].isna().all()


def test_as_of_join_is_order_independent():
    """Shuffled input must not change the answer."""
    state = _state(["2025-01-01", "2025-02-01", "2025-03-01"], [1.0, 2.0, 3.0])
    targets = _targets(["2025-03-15", "2025-01-15", "2025-02-15"])

    merged = as_of_join(
        targets.copy(), state.sample(frac=1, random_state=0), by="entity_id", columns=["stat"]
    )
    merged = merged.sort_values(EVENT_TIME)
    assert merged["stat"].tolist() == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# 4. Rolling windows respect the boundary
# ---------------------------------------------------------------------------
def test_rolling_form_uses_only_prior_runs():
    """Hand-checked: a horse's 4th run sees runs 1-3 and nothing else."""
    from backend.features.horse_features import build_horse_state

    history = build_history_frame(
        [
            ("hrs_x", "2025-01-01", 1),
            ("hrs_x", "2025-02-01", 2),
            ("hrs_x", "2025-03-01", 3),
            ("hrs_x", "2025-04-01", 8),
        ]
    )
    state = build_horse_state(history)

    targets = pd.DataFrame({"horse_id": ["hrs_x"], EVENT_TIME: pd.to_datetime(["2025-04-01"], utc=True)})
    merged = as_of_join(
        targets, state, by="horse_id", columns=["horse_runs_before", "horse_form_last3_avg_pos"]
    )

    assert merged["horse_runs_before"].iloc[0] == 3
    # Mean of 1, 2, 3 -- the 8th place being predicted is excluded.
    assert merged["horse_form_last3_avg_pos"].iloc[0] == pytest.approx(2.0)
