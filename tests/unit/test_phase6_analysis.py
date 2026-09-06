"""Baselines, the encompassing test, robustness segmentation and the report.

The encompassing test carries the weight here: it is the instrument that answers
"does the model know anything the price does not?", so it is checked against
constructed data where the true answer is known both ways.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from backend.ml.baselines import (
    FavouriteBaseline,
    MarketBaseline,
    ModelBaseline,
    RandomBaseline,
    UniformBaseline,
    compare_baselines,
    market_encompassing_test,
)
from backend.research.protocol import PROTOCOL, Verdict
from backend.research.validation_report import ValidationRun, render_markdown, run_validation
from backend.strategy.robustness import (
    MIN_SEGMENT_BETS,
    consistency_summary,
    render_segments,
    segment_analysis,
    segment_frame,
)
from backend.utils.exceptions import PipelineError

pytestmark = pytest.mark.unit


def make_market_frame(
    *,
    races: int = 300,
    runners: int = 8,
    model_skill: float = 0.0,
    model_noise: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Races with a known market and a model of controllable extra skill.

    Two knobs, and the distinction between them is the whole point of the
    encompassing test:

    ``model_skill``  moves the model towards the *truth* — real information the
                     price does not have.
    ``model_noise``  perturbs the model away from the market at random — a model
                     that looks different but knows nothing extra.

    A good test must separate them: noise alone must never register as signal.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for race_index in range(races):
        truth = rng.dirichlet(np.ones(runners) * 2)
        winner = rng.choice(runners, p=truth)

        # The market sees the truth blurred; the model sees it slightly less blurred.
        market = 0.75 * truth + 0.25 * rng.dirichlet(np.ones(runners))
        market = market / market.sum()
        model = (1 - model_skill) * market + model_skill * truth
        if model_noise:
            model = np.clip(model * np.exp(rng.normal(0, model_noise, runners)), 1e-6, None)
        model = model / model.sum()

        for runner_index in range(runners):
            rows.append(
                {
                    "race_id": f"rac_{race_index:05d}",
                    "horse_id": f"hrs_{race_index:05d}_{runner_index}",
                    "race_date": date(2025, 1, 1) + timedelta(days=race_index % 365),
                    "market_probability": float(market[runner_index]),
                    "model_probability": float(model[runner_index]),
                    "mkt_mean_odds": float(1.0 / (market[runner_index] * 1.15)),
                    "won": int(runner_index == winner),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "baseline", [RandomBaseline(), UniformBaseline(), FavouriteBaseline(), MarketBaseline(), ModelBaseline()]
)
def test_every_baseline_returns_a_race_distribution(baseline):
    frame = make_market_frame(races=40, seed=1)
    probabilities = baseline(frame)

    assert len(probabilities) == len(frame)
    totals = pd.DataFrame({"race_id": frame["race_id"], "p": probabilities}).groupby("race_id")["p"].sum()
    assert totals.round(6).eq(1.0).all()
    assert (probabilities >= 0).all()


def test_uniform_baseline_is_exactly_one_over_the_field():
    frame = make_market_frame(races=10, runners=8, seed=2)
    assert UniformBaseline()(frame) == pytest.approx(np.full(len(frame), 0.125))


def test_favourite_baseline_backs_the_shortest_price():
    frame = make_market_frame(races=20, seed=3)
    probabilities = FavouriteBaseline()(frame)

    scored = frame.assign(p=probabilities)
    for _, race in scored.groupby("race_id"):
        assert race.loc[race["p"].idxmax(), "mkt_mean_odds"] == race["mkt_mean_odds"].min()


def test_baselines_rank_in_the_expected_order():
    """Market beats favourite beats uniform beats random. If not, something is wrong."""
    frame = make_market_frame(races=600, seed=4)
    table = compare_baselines(frame).set_index("forecaster")

    assert table.loc["market", "race_log_loss"] < table.loc["uniform", "race_log_loss"]
    assert table.loc["uniform", "race_log_loss"] < table.loc["random", "race_log_loss"]
    assert table.loc["market", "top1_hit_rate"] > table.loc["random", "top1_hit_rate"]


def test_baseline_comparison_on_an_empty_frame():
    assert compare_baselines(pd.DataFrame()).empty


def test_baselines_fall_back_when_their_column_is_missing():
    frame = make_market_frame(races=20, seed=5).drop(columns=["market_probability", "mkt_mean_odds"])
    assert MarketBaseline()(frame) == pytest.approx(UniformBaseline()(frame))
    assert FavouriteBaseline()(frame) == pytest.approx(UniformBaseline()(frame))


# ---------------------------------------------------------------------------
# Encompassing — the decisive instrument
# ---------------------------------------------------------------------------
def test_a_model_identical_to_the_market_is_flagged_as_collinear():
    """A model that reproduces the price exactly is encompassed by definition.

    The regression cannot run — the design is singular — so this must be
    reported as encompassing rather than as a convergence failure.
    """
    frame = make_market_frame(races=800, model_skill=0.0, seed=6)
    result = market_encompassing_test(frame)

    assert result.collinear
    assert result.market_encompasses_model
    assert not result.model_adds_information
    assert "indistinguishable from the price" in result.verdict


def test_a_model_that_differs_only_by_noise_adds_nothing():
    """The realistic null: visibly different from the price, but knowing no more.

    This is the case that separates a real instrument from a credulous one. The
    model's forecasts are far from identical to the market's, so nothing is
    collinear and the regression runs — and it must still conclude that the
    extra variation is worthless.
    """
    frame = make_market_frame(races=1500, model_skill=0.0, model_noise=0.35, seed=61)
    result = market_encompassing_test(frame)

    assert result.converged
    assert not result.collinear
    assert abs(result.forecast_correlation) < 0.99
    assert not result.model_adds_information
    assert result.market_encompasses_model


def test_a_genuinely_better_model_is_detected():
    frame = make_market_frame(races=800, model_skill=0.5, seed=7)
    result = market_encompassing_test(frame)

    assert result.converged
    assert result.model_adds_information
    assert result.model_coefficient > 0
    assert result.model_z >= 2.0


def test_combining_forecasts_cannot_worsen_the_fit():
    frame = make_market_frame(races=800, model_skill=0.4, seed=8)
    result = market_encompassing_test(frame)
    assert result.log_loss_combined <= result.log_loss_market_only + 1e-9
    assert result.likelihood_ratio >= 0


def test_encompassing_needs_enough_data():
    assert not market_encompassing_test(make_market_frame(races=5, seed=9)).converged
    assert not market_encompassing_test(pd.DataFrame()).converged


def test_encompassing_needs_both_columns():
    frame = make_market_frame(races=200, seed=10).drop(columns=["market_probability"])
    assert not market_encompassing_test(frame).converged


def test_encompassing_result_serialises():
    result = market_encompassing_test(make_market_frame(races=400, model_skill=0.4, seed=11))
    payload = result.as_dict()
    assert "model_adds_information" in payload
    assert "verdict" in payload


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------
def make_ledger(
    *, bets: int = 600, roi_by_year: dict[int, float] | None = None, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    roi_by_year = roi_by_year or {2023: 0.10, 2024: 0.10}
    years = list(roi_by_year)

    rows = []
    for index in range(bets):
        year = years[index % len(years)]
        odds = float(rng.uniform(2.0, 9.0))
        # Choose a win probability that delivers the requested ROI in expectation.
        probability = min(0.95, (1 + roi_by_year[year]) / odds)
        won = int(rng.random() < probability)
        stake = 10.0
        profit = stake * (odds - 1) if won else -stake
        rows.append(
            {
                "race_id": f"rac_{index:05d}",
                "horse_id": f"hrs_{index:05d}",
                "race_date": date(year, 1 + index % 12, 1 + index % 28),
                "odds": odds,
                "stake": stake,
                "won": won,
                "profit_loss": profit,
                "bankroll_after": 10000.0,
                "model_probability": probability,
            }
        )
    return pd.DataFrame(rows)


def make_context(ledger: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "race_id": ledger["race_id"],
            "race_type": rng.choice(["Flat", "Chase", "Hurdle"], len(ledger)),
            "race_is_handicap": rng.integers(0, 2, len(ledger)),
            "race_field_size": rng.integers(5, 20, len(ledger)),
        }
    )


def test_segments_split_along_every_dimension():
    ledger = make_ledger(bets=800, seed=1)
    results = segment_analysis(ledger, make_context(ledger, seed=1))

    dimensions = {result.dimension for result in results}
    assert {"year", "code", "handicap", "odds_band", "field_size_band"} <= dimensions


def test_year_segments_recover_the_roi_they_were_built_with():
    ledger = make_ledger(bets=4000, roi_by_year={2023: 0.20, 2024: -0.20}, seed=2)
    years = {r.segment: r for r in segment_analysis(ledger) if r.dimension == "year"}

    assert years["2023"].roi > 0
    assert years["2024"].roi < 0


def test_thin_segments_are_reported_but_not_counted():
    ledger = make_ledger(bets=40, seed=3)
    results = segment_analysis(ledger)

    assert results
    assert all(not result.is_reportable for result in results if result.bets < MIN_SEGMENT_BETS)
    assert consistency_summary(results).reportable_segments == 0


def test_consistency_counts_profitable_years():
    ledger = make_ledger(bets=4000, roi_by_year={2022: 0.15, 2023: 0.15, 2024: -0.15}, seed=4)
    summary = consistency_summary(segment_analysis(ledger))

    assert summary.years_covered == 3
    assert summary.profitable_years == 2
    assert summary.profitable_year_fraction == pytest.approx(2 / 3)
    assert summary.worst_year_roi < 0 < summary.best_year_roi


def test_the_corrected_threshold_is_stricter_than_the_nominal_one():
    ledger = make_ledger(bets=3000, seed=5)
    summary = consistency_summary(segment_analysis(ledger, make_context(ledger, seed=5)))

    assert summary.corrected_threshold > 2.0
    assert len(summary.passing_corrected) <= len(summary.passing_nominal)


def test_segment_helpers_handle_an_empty_ledger():
    assert segment_analysis(pd.DataFrame()) == []
    assert segment_frame([]).empty
    assert "(no segments)" in render_segments([])


def test_segment_rendering_and_serialisation():
    ledger = make_ledger(bets=600, seed=6)
    results = segment_analysis(ledger)

    assert "segment" in render_segments(results, dimension="year")
    frame = segment_frame(results)
    assert {"dimension", "segment", "bets", "roi", "t_statistic"} <= set(frame.columns)


def test_consistency_renders():
    summary = consistency_summary(segment_analysis(make_ledger(bets=600, seed=7)))
    assert "Profitable years" in summary.render()
    assert "corrected_threshold" in summary.as_dict()


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------
def test_validation_refuses_an_unready_dataset(db_session):
    """The gate that stops a synthetic report from ever being produced."""
    from backend.research.synthetic import SyntheticConfig, generate_synthetic_data

    generate_synthetic_data(db_session, SyntheticConfig(n_days=60, races_per_day=2, seed=8))
    with pytest.raises(PipelineError, match="pre-flight gate"):
        run_validation(db_session)


def test_validation_on_an_empty_database_is_refused(db_session):
    with pytest.raises(PipelineError, match="pre-flight gate"):
        run_validation(db_session)


def test_report_renders_the_verdict_and_every_section():
    from backend.research.audit import DatasetAudit
    from backend.research.protocol import VerdictInputs

    audit = DatasetAudit(source="real", races=1000)
    run = ValidationRun(protocol=PROTOCOL, audit=audit)
    run.verdict = PROTOCOL.verdict(
        VerdictInputs(bets=500, roi=0.05, t_statistic=2.5, profitable_year_fraction=0.8, years_covered=5)
    )

    markdown = render_markdown(run)
    for expected in (
        "# Phase 6 — final validation report",
        "## 0. Pre-flight gates",
        "## 8. Final verdict",
        "### Limitations",
    ):
        assert expected in markdown
    assert run.verdict.verdict is Verdict.EDGE_CONFIRMED
    assert "EDGE_CONFIRMED" in markdown, "the report must print the specified label"


def test_report_flags_a_harness_run():
    from backend.research.audit import DatasetAudit

    run = ValidationRun(protocol=PROTOCOL, audit=DatasetAudit(source="synthetic"))
    run.notes.append("RUN ON A DATASET THAT FAILED A PRE-FLIGHT GATE")
    assert "⚠️" in render_markdown(run)
