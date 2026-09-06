# Horse Quant Platform

[![CI](https://github.com/pranay123-stack/horse-quant-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/pranay123-stack/horse-quant-platform/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-1%2C081%20passing-brightgreen)](tests/)
[![Coverage gate](https://img.shields.io/badge/coverage%20gate-90%25-brightgreen)](.github/workflows/ci.yml)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Docker](https://img.shields.io/badge/Docker-compose-2496ED?logo=docker&logoColor=white)](docker-compose.yml)

A quantitative betting research and execution platform for **UK horse racing**, built
with the same discipline as a systematic trading system: reproducible data, versioned
models, an explicit edge criterion, and honest backtests.

The thesis is simple. A bookmaker's price implies a probability. If our model's
probability is meaningfully higher than the price implies, the bet has positive
expected value:

```
EV = (p_model × decimal_odds) − 1
```

Everything in this repository exists to make `p_model` trustworthy, to measure the
edge without fooling ourselves, and to size stakes so that a real edge survives
variance.

---

## The 30-second version

| | |
|---|---|
| **What it is** | End-to-end quant stack: ingest → point-in-time features → calibrated probability model → expected-value strategy → walk-forward backtest → live daily signals → API + dashboard |
| **Scale** | ~23,400 lines of application code, ~13,200 lines of tests, 162 Python modules, 89 point-in-time features, 6 CLIs |
| **Tests** | **1,081 passing** (73 integration tests skip without PostgreSQL); CI runs lint + `mypy` + tests behind a **90% coverage gate**, applies and rolls back every migration, and smoke-tests the Docker image |
| **Stack** | Python 3.12 · FastAPI · SQLAlchemy 2.0 · Alembic · PostgreSQL 16 · pandas / scikit-learn / XGBoost / LightGBM · Docker · GitHub Actions |
| **Honest status** | The engine is complete and tested. **The edge is unproven** — the Racing API subscription is inactive, so every number in this README comes from synthetic data and says nothing about real markets. Section [Real data validation](#real-data-validation) explains exactly what is blocked and why |

### Why this repository is worth ten minutes

Most betting repositories show a model with a flattering backtest. The
engineering that matters is the part that stops a backtest from flattering you,
and that is what this one is mostly made of:

- **Leakage is structurally impossible, not merely avoided.** Every feature is
  attached by one `merge_asof` helper with `allow_exact_matches=False`. A test
  builds features, reveals later races, rebuilds, and asserts nothing earlier
  moved — [it caught a real leak in this codebase](#testing-that-the-guarantee-holds),
  a median imputation computed across the whole dataset.
- **The analysis plan was frozen and content-hashed before any data existed.**
  One pre-registered primary hypothesis, a mechanical four-way verdict function,
  and twelve pre-declared secondary tests that drag the Bonferroni threshold from
  t = 2.0 to **2.89**. Editing the plan after seeing results changes the
  fingerprint and fails the run.
- **The client PDF cannot call the strategy profitable** unless the frozen
  decision rule returns `EDGE_CONFIRMED` — asserted by reading the generated
  PDF's text back and checking the encouraging phrases are absent.
- **Negative controls are wired in and they pass.** Betting the market's own
  margin-free probability places exactly zero bets, as it must. Strip the market
  features and the model loses money — reported plainly, because it means the
  edge is *market denoising*, not independent alpha.
- **Live and backtest are proven to be the same code**, not assumed to be:
  `make replay day=…` re-scores a past day through the live path and fails on any
  divergence.
- **The uncomfortable results are in the README.** Calibration made log loss
  worse and it says so. An unfiltered EV rule turns into a longshot-betting
  machine and there is a table showing it.

<sub>The willingness to publish the failures is the point. A platform that can only
produce good news is not a research platform.</sub>

---

## Contents

- [The product](#the-product) · [Status](#status) · [Architecture](#architecture)
- [Quick start](#quick-start) · [The Racing API](#the-racing-api) · [Ingesting data](#ingesting-data) · [Data model](#data-model) · [HTTP API](#http-api)
- [Quantitative research layer](#quantitative-research-layer) — leakage guarantee, 89 features, data quality, datasets, backtesting
- [The probability engine](#the-probability-engine) — four-window splits, calibration, race normalisation, model registry
- [Value betting](#value-betting) — walk-forward, strategy comparison, market dependency, cost sensitivity
- [How the product works](#how-the-product-works) — daily signals, endpoints, settled performance
- [Real data validation](#real-data-validation) — frozen protocol, pre-flight gates, encompassing regression, reports
- [Configuration](#configuration) · [Logging](#logging) · [Database](#database) · [Testing](#testing) · [Development commands](#development-commands) · [Responsible use](#responsible-use)

---

## The product

Train once on history, then run one command each morning:

```bash
make train-production   # history -> LR / LightGBM / XGBoost -> promoted champion
make predictions        # fetch today's card, score it, store the signals
make dashboard          # http://localhost:8000/dashboard/
```

What a user sees:

```
Race                  Horse                Prob   Odds     EV   Recommendation
Kempton 18:30         Thunder King          32%   7.00   +124%  BET
Kempton 18:30         Silver Lining         18%   4.50    -19%  NO_BET
```

A runner is backed only when **EV > 5% and probability > 10%**, with a real
price between 1.01 and 50.0 in a field of five or more. Everything else is
labelled `NO_BET` *with the reason*, because "no price available" and "expected
value too low" send a person to completely different places.

---

## Status

| Phase | Scope | State |
|------:|-------|-------|
| 1 | Project setup & architecture | **Complete** |
| 2 | The Racing API integration | **Complete** |
| 3 | Data quality, features, research datasets, backtest foundation | **Complete** |
| 4 | Probability engine (LR / XGBoost / LightGBM + calibration) | **Complete** |
| 5 | Odds integration, value betting, walk-forward backtest | **Complete** |
| 6 | Real-data validation: audit, protocol, baselines, verdict | **Built — blocked on the API subscription** |
| 7 | **The product**: live predictions, betting signals, dashboard | **Complete** |
| 8 | Backtesting: reporting, sweeps, walk-forward aggregation | Planned |
| 9 | Dashboard | Planned |
| 10 | Production hardening (auth, Celery, Redis) | Planned |

---

## Architecture

Dependencies point in one direction only — downward. Nothing in `utils` or
`database` knows the HTTP layer exists, which is what keeps the ETL, training and
backtest code runnable from a script, a notebook or a Celery worker.

```
      ┌──────────────────┐
      │  The Racing API  │
      └────────┬─────────┘
               │  services/racing_api/      auth · retry · rate limit   (Phase 2)
      ┌────────▼─────────┐
      │  data_pipeline   │  validate → parse → idempotent upsert        (Phase 2)
      └────────┬─────────┘
               │
      ┌────────▼─────────┐
      │   PostgreSQL     │  races · race_runners · race_results
      │                  │  horses · jockeys · trainers · odds_history
      └────────┬─────────┘  race_features
               │
      ┌────────▼─────────┐
      │  data_quality    │  which races are trustworthy?               (Phase 3)
      └────────┬─────────┘
               │
      ┌────────▼─────────┐
      │    features      │  89 point-in-time features
      │                  │  as_of_join: leakage structurally impossible (Phase 3)
      └────────┬─────────┘
               │
      ┌────────▼─────────┐
      │    research      │  dataset builder · date-based splits         (Phase 3)
      └────────┬─────────┘
               │
      ┌────────▼─────────────────────────────────────────────┐
      │                        ml                            │  (Phase 4)
      │  TrainingDatasetBuilder   train/valid_stop/          │
      │                           valid_calib/test           │
      │  FeaturePreprocessor      impute · scale · one-hot   │
      │  RacingModel              LR │ XGBoost │ LightGBM    │
      │  ProbabilityCalibrator    Platt │ isotonic           │
      │  normalise_within_race    one winner ⇒ Σp = 1        │
      │  ModelRegistry            versioned · promotable     │
      └────────┬─────────────────────────────────────────────┘
               │  calibrated win probabilities
               │
      ┌────────▼─────────────────────────────────────────────┐
      │                     strategy                         │  (Phase 5)
      │  odds.py            consensus price → fair prob      │
      │                     best price      → bet price      │
      │  expected_value.py  EV = p·odds − 1,  edge           │
      │  filters.py         where the model may be trusted   │
      │  staking.py         flat │ fractional Kelly + limits │
      │  backtester.py      walk-forward: retrain → settle   │
      │  reports.py         ROI · drawdown · t-statistic     │
      └────────┬─────────────────────────────────────────────┘
               │  bet_ledger
      ┌────────▼─────────────────────────────────────────────┐
      │                    validation                        │  (Phase 6)
      │  research/protocol.py   the plan, frozen and hashed  │
      │                         BEFORE the data exists       │
      │  research/audit.py      data quality + timestamps    │
      │  research/preflight.py  8 gates, stop at the first   │
      │                         that fails                   │
      │  research/execution.py  7 steps, checkpointed and    │
      │                         resumable                    │
      │  ml/baselines.py        encompassing: does the model │
      │                         beat the price, conditional  │
      │                         on knowing the price?        │
      │  strategy/robustness.py segments + Bonferroni        │
      │  research/validation_   ONE verdict, computed by the │
      │            report.py    frozen rule, not written     │
      │  research/client_       the PDF — cannot claim       │
      │            report.py    profit without the verdict   │
      │  research/readiness.py  API/Data/Model/Backtest/Risk │
      └────────┬─────────────────────────────────────────────┘
               │  Phase6_Final_Report.md · the client PDF
               │  PRODUCTION_READINESS.md · research_runs
               │
      ┌────────▼─────────────────────────────────────────────┐
      │                prediction_service                    │  (Phase 7)
      │  training.py   history -> LR/LGBM/XGB -> champion    │
      │  service.py    today's card -> probabilities -> EV   │
      │  rules.py      EV>5% AND p>10% -> BET / NO_BET       │
      │  daily.py      the morning job, and settlement       │
      │  replay.py     proves live == backtest               │
      │  performance.py  the settled record, with its t      │
      └────────┬─────────────────────────────────────────────┘
               │  predictions table
      ┌────────▼─────────┐
      │  api (FastAPI)   │  → frontend/dashboard
      └──────────────────┘
```

### Layout

```
horse_quant_platform/
├── backend/
│   ├── api/                 HTTP layer
│   │   ├── v1/              health, meta, races, horses, odds, predictions
│   │   ├── deps.py          FastAPI dependencies
│   │   ├── errors.py        exception → HTTP mapping
│   │   └── middleware.py    request ids + access logging
│   ├── backtesting/         engine · metrics · baseline predictors
│   ├── database/            engine, sessions, declarative base
│   ├── data_pipeline/       importer.py · report.py
│   │                        backfill.py  resumable + audited (backfill_runs)
│   ├── data_quality/        rules · validator · report
│   ├── features/            point-in-time feature engineering
│   ├── ml/                  the probability engine
│   │   ├── dataset.py       train/valid_stop/valid_calib/test, by date
│   │   ├── preprocessing/   feature schema + model matrix
│   │   ├── models/          LR · XGBoost · LightGBM · registry
│   │   ├── calibration/     Platt · isotonic · before/after evidence
│   │   ├── metrics/         classification + race-level
│   │   ├── baselines.py     random · uniform · favourite · market + encompassing
│   │   ├── train.py         orchestration
│   │   ├── evaluate.py      reports and comparison
│   │   └── predict.py       race in, probabilities out
│   ├── models/              ORM: entities · racing · features · operations
│   │                        predictions.py  the product's stored output
│   ├── prediction_service/  THE PRODUCT
│   │                        training.py    production model + manifest
│   │                        service.py     racecards -> betting signals
│   │                        rules.py       BET / NO_BET, and why
│   │                        daily.py       the morning job + settlement
│   │                        replay.py      live == backtest
│   │                        performance.py the settled record
│   ├── research/            dataset builder · time splits · synthetic data
│   │                        protocol.py   the frozen, fingerprinted analysis plan
│   │                        audit.py      data quality + timestamp consistency
│   │                        preflight.py  the gates that stop a bad run
│   │                        execution.py  the 7-step resumable workflow
│   │                        validation_report.py  the Phase 6 verdict
│   │                        client_report.py      the client PDF
│   │                        readiness.py          PASS/FAIL per component
│   ├── schemas/             read models for our own API
│   ├── services/racing_api/ client · auth · parsers · schemas · rate limit
│   │                        capabilities.py  what the plan can actually do
│   ├── strategy/            odds · EV · filters · staking · backtest · reports
│   │                        robustness.py  segment analysis with multiplicity
│   ├── utils/               config, logging, exceptions, time helpers
│   └── main.py              FastAPI application factory
├── frontend/dashboard/      index.html — today's card, performance, model
├── migrations/              Alembic
├── scripts/racing_cli.py    check · racecards · results · backfill
├── scripts/research_cli.py  quality · features · dataset · backtest · synthetic
├── scripts/ml_cli.py        train · compare · importance · predict · registry
├── scripts/strategy_cli.py  predictions · backtest · compare · market-dependency
├── scripts/predict_cli.py   train · today · daily · performance · replay
├── scripts/phase6_cli.py    protocol · api-status · phase6-preflight
│                            · phase6-run · steps · experiments · audit
├── tests/{unit,integration,fixtures}
├── logs/                    api.log · pipeline.log · model.log · error.log
├── models/                  serialised model artefacts (joblib)
├── data/{raw,processed}     cached API payloads and feature sets
├── reports/                 REAL_DATA_AUDIT.md · Phase6_Final_Report.md
│                            · Horse_Racing_Quant_Strategy_Report.pdf
│                            · PRODUCTION_READINESS.md
├── docker-compose.yml       postgres · redis · api · adminer
├── Dockerfile               multi-stage, non-root
├── Makefile                 the entire developer workflow
├── requirements.txt         runtime dependencies
├── requirements-dev.txt     + testing and quality tooling
├── pyproject.toml           pytest / ruff / mypy / coverage config
├── .env.example             every setting, documented
└── PHASE6_STATUS.md         what is built, what is blocked, and why
```

---

## Quick start

Requires **Python 3.12** and **Docker**.

```bash
cd horse_quant_platform

# 1. virtualenv + dependencies + .env
make setup

# 2. fill in credentials (Racing API, database password)
$EDITOR .env

# 3. start PostgreSQL and Redis
make db-up

# 4. run the test suite
make test

# 5. run the API
make run
```

Then:

| URL | What |
|-----|------|
| http://localhost:8000/ | service banner |
| http://localhost:8000/docs | interactive OpenAPI docs |
| http://localhost:8000/health | liveness |
| http://localhost:8000/health/ready | readiness (DB + credentials) |
| http://localhost:8000/api/v1/meta/config | effective configuration, secrets masked |

Full stack in containers instead:

```bash
make build && make up && make logs
```

> **Port conflicts.** If you already run PostgreSQL or Redis locally, `make db-up`
> will fail to bind. Set free host ports in `.env` — `POSTGRES_PORT=5462`,
> `REDIS_PORT=6380` — and compose will publish there instead. The containers
> always talk to each other on the standard ports internally.

---

## The Racing API

The integration lives in [`backend/services/racing_api/`](backend/services/racing_api/):

| Module | Responsibility |
|--------|----------------|
| `client.py` | async httpx client — pooling, timeouts, retries, pagination |
| `authentication.py` | HTTP Basic credentials, held as `SecretStr` end to end |
| `exceptions.py` | error taxonomy, including the four distinct 401s |
| `parsers.py` | string payloads → typed domain objects |
| `schemas.py` | wire contracts (lenient) and domain contracts (strict) |
| `rate_limit.py` | async token bucket |

```python
from backend.services.racing_api import RacingAPIClient

async with RacingAPIClient() as client:
    races = await client.get_races(date="2026-08-10", region_codes=["gb", "ire"])
    odds = await client.get_odds(races[0].race_id, races[0].runners[0].horse.horse_id)
```

### Check your credentials first

```bash
python -m scripts.racing_cli check
```

This exists because the API returns **`401` for four different reasons**, and the
remedies are completely different:

| Response body | Meaning | What to do |
|---------------|---------|------------|
| `Not authenticated` | no credentials sent | set `RACING_API_*` in `.env` |
| `Incorrect username` / `Incorrect password` | credentials wrong | fix the credentials |
| `Subscription inactive` | **account valid, plan not active** | renew at theracingapi.com |

Each maps to its own exception (`RacingAPIAuthenticationError` vs
`SubscriptionInactiveError`), so an inactive subscription never sends anyone
hunting for a typo in a password that is demonstrably correct.

### Two things about this API that shape the code

**Everything is a string.** `age`, `lbs`, `ofr`, `rpr`, `draw` and decimal odds
all arrive as JSON strings, and missing values are `""`, `"-"` or `"–"` rather
than `null`. `int(runner["ofr"])` therefore raises on perfectly ordinary rows.
Every coercion in `parsers.py` is total — it returns `None` instead of raising,
because one odd runner must never abort a whole race day.

**Racecards and results name the same things differently:**

| Concept | Racecards | Results |
|---------|-----------|---------|
| class | `race_class` | `class` |
| distance | `distance`, `distance_f` | `dist`, `dist_y`, `dist_m`, `dist_f` |
| off time | `off_time` | `off` |
| official rating | `ofr` | `or` |
| sex restriction | `sex_restriction` | `sex_rest` |

Both are normalised onto one canonical `RaceSchema`.

Retries cover transport errors, timeouts, `429` and `5xx` with exponential
backoff and full jitter, honouring `Retry-After`. `401`, `403` and `404` are
**never** retried — presenting bad credentials repeatedly burns quota and risks
tripping abuse protection.

---

## Ingesting data

```bash
# today's racecards (UK + Ireland)
python -m scripts.racing_cli racecards

# a specific day, with odds included (pro tier)
python -m scripts.racing_cli racecards --date 2026-08-10 --tier pro

# finished races for a range — the training labels
python -m scripts.racing_cli results --start 2026-08-01 --end 2026-08-07

# long historical backfill, one day at a time (resilient)
python -m scripts.racing_cli backfill --start 2025-08-01 --end 2026-08-01
```

Every run prints a summary:

```
Import summary
--------------
  Races seen      : 520
  Imported        : 500
  Updated         : 0
  Skipped (dupes) : 20
  Runners         : 4310 new / 0 updated
  Results         : 0 new / 0 updated
  Odds quotes     : 8620 new / 0 unchanged
  Failed          : 2
      - rac_18f2a [persist] IntegrityError: value too long for type character varying(40)
```

Two properties make this safe to schedule:

* **Idempotent.** Every write is an upsert keyed on the Racing API's own
  identifiers, so re-running a day changes only what genuinely changed. Odds are
  keyed on `(race, horse, bookmaker, recorded_at)`, so an unchanged price is
  skipped while a real move is appended as history.
* **Failure-isolated.** Each race is written inside its own `SAVEPOINT`. A
  malformed race rolls back alone, is recorded in the summary with enough detail
  to retry, and the remaining races on the card still land.

Updates never overwrite a populated column with `NULL` — racecards and results
carry different subsets of a horse's attributes, and a blind overwrite would
erase a horse's colour every time a result was imported.

---

## Data model

```
courses ──< races ──< race_runners  >── horses
              │  ├──< race_results  >── horses · jockeys · trainers
              │  ├──< odds_history  >── horses
              ├──< race_features    >── horses      (Phase 3)
              └──< bet_ledger       >── horses      (Phase 5)
```

`race_features` stores one point-in-time vector per runner per race: the eight
composite scores as real columns (stable, queryable, dashboard-ready) plus the
full evolving feature set as `JSONB`, so adding a feature needs no migration.
`feature_version` records which code produced each row — vectors from different
versions must never be mixed into one training set.

`race_runners` (the declaration) and `race_results` (the outcome) are separate
tables on purpose. It is the structural defence against look-ahead bias: a
feature query that joins only `race_runners` **cannot** accidentally read a
finishing position.

`finishing_position` is `NULL` for non-finishers, with `finishing_status`
recording why (`pulled_up`, `fell`, `unseated_rider`, `brought_down`…). Treating
a faller as "did not win" is correct; treating it as "finished last" is not.

Money and odds are `NUMERIC`, never `float`.

---

## HTTP API

```bash
# races, newest first, with filters
curl 'localhost:8000/api/v1/races?race_date=2026-08-10&region=gb&limit=20'

# one race: full card, plus results once run
curl localhost:8000/api/v1/races/rac_18f2a

# horse profile with recent form and career strike rate
curl 'localhost:8000/api/v1/horses/hrs_9c11?runs=10'

# the market: best price per runner, plus the overround
curl localhost:8000/api/v1/odds/rac_18f2a

# full price history for one runner
curl localhost:8000/api/v1/odds/rac_18f2a/hrs_9c11
```

The odds endpoint returns an **overround** — the sum of implied probabilities at
the best available price. Above `1.0` is the bookmaker's margin; below `1.0`
means every runner could be backed at a profit, which is logged as a warning
because it usually means stale prices rather than free money. Phase 7 divides by
the overround to recover the normalised market probability the model must beat.

Interactive docs: <http://localhost:8000/docs>

---

## Quantitative research layer

Phase 3 turns stored racing data into model-ready features without letting the
future leak into the past. Three modules, in the order data flows through them:

```
races/results/odds  ->  data_quality  ->  features  ->  research  ->  backtesting
                        (which races      (89 point-   (dataset,     (simulate,
                         are usable?)      in-time      time-based    measure)
                                           features)    splits)
```

### The one guarantee everything rests on

A feature for a race at time `T` may use **only** information that existed before
`T`. This is enforced structurally, not by discipline. Every feature is attached
by a single function, [`as_of_join`](backend/features/base.py):

```python
pd.merge_asof(targets, state, on="event_time", by="horse_id",
              direction="backward", allow_exact_matches=False)
```

For a race at `T` this selects the last history row **strictly before** `T`,
whose cumulative state summarises every prior event and no later one. There is no
code path by which a later row can be chosen, so leakage cannot be introduced by
forgetting a filter — only by deliberately bypassing this function.

`allow_exact_matches=False` is the load-bearing detail: it excludes the race's own
result from its own features.

### Testing that the guarantee holds

[`tests/unit/test_no_leakage.py`](tests/unit/test_no_leakage.py) attacks it from
four directions. The decisive one is **future injection**: build features, make
later races visible, rebuild, and assert every earlier value is unchanged.

That test earned its keep immediately — it caught a real leak in this codebase.
`horse_consistency` filled missing values with `spread.median()`, a median taken
across the *whole dataset* including future races, so adding later races silently
changed earlier features. It never raised, it never looked wrong, and it would
have flattered every backtest built on it.

### Features

89 features across six groups. Full list with live coverage statistics:
`make features manifest=1`.

| Group | n | Examples |
|-------|--:|----------|
| **horse** | 31 | `horse_form_last3_avg_pos`, `horse_avg_speed`, `horse_speed_trend`, `horse_consistency`, `horse_distance_suitability`, `horse_going_suitability`, `horse_class_move`, `horse_days_since_prev_run` |
| **jockey** | 9 | `jky_strike_rate` (last 30 rides), `jky_roi`, `jky_course_strike_rate` |
| **trainer** | 11 | `trn_win_rate_30d`, `trn_roi_30d`, `trn_win_rate_window` (100 runners), `trn_jky_win_rate` |
| **race** | 17 | `race_distance_furlongs`, `race_going_value`, `runner_or_rank`, `runner_weight_vs_field`, `runner_draw_pct` |
| **market** | 13 | `mkt_open_odds`, `mkt_latest_odds`, `mkt_odds_change_pct`, `mkt_implied_prob_norm`, `mkt_overround`, `mkt_rank` |
| **score** | 8 | `horse_form_score`, `speed_score`, `jockey_score`, `trainer_score`, `market_score`, `distance_score`, `going_score`, `class_score` |

Three deliberate choices worth knowing about:

**Small samples are shrunk toward a prior.** A horse that won its only run over a
trip is not a 100% distance specialist. Every rate is empirical-Bayes shrunk
(`k=6`) toward the population base rate, so small samples say what they actually
say: not much.

**Scores are percentile ranks *within the race*.** A speed figure of 95 is
excellent in a Class 6 seller and moderate in a Group race. Racing is a ranking
problem, so most of the signal is cross-sectional, and ranking also puts every
component on a common `[0, 1]` scale before they are combined.

**Missing is missing.** A runner with no captured market gets `NULL` odds, never
an imputed price — a fabricated price looks to the strategy engine like a real
betting opportunity.

The composite score follows the Phase 3 specification exactly:

```
horse_form_score = 0.30·recent_form + 0.25·speed + 0.20·distance_fit
                 + 0.15·going_fit  + 0.10·class_rating
```

### Data quality

```bash
make quality
```

```
Data quality report
-------------------
  Total races     : 720
  Valid           : 720
  Rejected        : 0
  Pass rate       : 100.00%
```

Rules live in [`backend/data_quality/rules.py`](backend/data_quality/rules.py),
one pure function per check, so every threshold is independently arguable. The
`ERROR` / `WARNING` split has teeth: `ERROR` excludes the race from research,
`WARNING` keeps it and flags it. Missing race class is only a warning — most UK
jumps races have none, and rejecting them would skew the sample towards the Flat.

### Building a dataset

```bash
make dataset out=data/processed/train.parquet
# or, with a time-based split written alongside it:
python -m scripts.research_cli dataset --out data/processed/train.parquet --train-end 2025-05-31
```

```
Research dataset
----------------
  Rows            : 5760
  Races           : 720
  Features        : 90
  Target          : won (base rate 12.500%)
  Date range      : 2025-01-01 → 2025-06-29
  Excluded races  : 0 (failed data quality)
```

**Splits are always by date, never random.** A random split puts races from the
same day — often the same race — on both sides, and form features on the training
side are built from races sitting in the test set. Both inflate measured accuracy
dramatically, and neither looks like a bug: the model simply appears good and then
loses money. `time_split()` and `walk_forward_splits()` split on the calendar and
`assert_no_leakage_between()` fails the build if a boundary is ever violated.

An `embargo_days` gap after the training cutoff removes the residual overlap from
rolling windows that span the boundary.

### Backtesting

```bash
make backtest predictor=all
```

```
 predictor  bets  strike_rate     roi  profit  max_drawdown_pct
    market     0       0.0000  0.0000     0.0            0.0000
   uniform   640       0.0109 -0.7825 -5007.8            0.5179
form_score   639       0.1440 -0.3181 -2032.6            0.2151
```

Read this table carefully, because it is the most instructive output in the phase:

* **`market` places zero bets.** It must. Its probabilities come from the same
  prices it is offered, so its EV is exactly `1/overround − 1`, always negative.
  A framework that let this strategy bet — or profit — would be broken. This is
  the negative control.
* **`uniform` bets 640 times at average odds of 27.2 and an average "EV" of
  +240%.** It loses 78%. A flat `1/8` probability applied to a 27/1 shot claims
  an enormous edge; the true chance is nearer 1%. **Uncalibrated probabilities
  manufacture spectacular fake edges, and an EV filter turns straight into a
  longshot-betting machine.**
* **`form_score` strikes at 14.4% against a 12.5% base rate** — so the composite
  score genuinely carries signal — and still loses 32%, because an ordinal score
  is not a probability.

That is the case for Phase 6 in three lines: the ranking is useful, the
calibration is missing, and only calibration turns one into the other.

The engine is deliberately pessimistic. Its assumptions are listed in
[`backend/backtesting/engine.py`](backend/backtesting/engine.py) and every one
makes live trading harder than the simulation. In particular it will **not** bet
at starting price: SP is only determined at the off and reaches us through the
results feed, so betting at it is look-ahead. (`allow_starting_price=True` exists
for research, and labels the result unrealisable.)

### Working without live data

The Racing API subscription is inactive, so there is no real data to build on.
[`backend/research/synthetic.py`](backend/research/synthetic.py) generates a
deterministic synthetic season — plausible form, correlated ratings, a market
with a configurable overround — purely so the pipeline can be exercised and
tested end to end.

```bash
make synthetic days=180
```

**Nothing produced by it says anything about real betting markets.** A model that
scores well on synthetic data has learned the generator. It exists to prove the
plumbing and to give the leakage tests a world where the answer is known by
construction.

---

## The probability engine

Phase 4 answers one question: *for each horse in a race, what is the probability
it wins?* No odds are consulted and no bet is suggested — that is Phase 5.

```bash
make train detail=1        # train, calibrate, evaluate, register
make compare               # comparison table from the registry
make importance            # what the champion actually uses
make predict race=rac_18f2a
```

### A race is not eight independent coin flips

Exactly one runner wins, so their probabilities must sum to one. A per-runner
binary classifier does not know that: ask it about eight runners and the answers
might total 0.7 or 1.4. Every prediction is therefore **normalised within its
race**, after calibration and before use.

The order matters. Calibration is a pointwise map learned from (score, outcome)
pairs; feeding it already-normalised values would mean fitting it on numbers
whose scale depends on how many horses happened to line up.

It also changes how models are scored. Per-runner log loss is dominated by the
seven easy negatives in an eight-runner field — "nobody wins" scores well. **Race
log loss**, `-mean(log p_winner)`, cannot be gamed that way, and it has a
meaningful reference point: the uniform model, `log(field_size)` ≈ 2.08. That is
the default criterion for choosing a champion.

### Four windows, not three

| Window | Job |
|--------|-----|
| `train` | fit the estimator *and* the preprocessor |
| `valid_stop` | early stopping — the model is *chosen* here |
| `valid_calib` | fit the probability calibrator |
| `test` | scored once, at the end |

Splitting validation in two is not fussiness. A calibrator fitted on the rows
that chose the stopping point sees predictions that are already flattering on
those rows, and learns a correction that is too gentle. The halves are split
chronologically, so the calibration slice is also the more recent one.

Default windows follow the specification — train 2018–2023, validate 2024, test
2025–2026 — with a 14-day embargo after the training cutoff, because rolling
features spanning the boundary share history with training.

### Results

Champion selection is by race log loss on the untouched test window.

```
                       model  log_loss   brier  roc_auc     ece  race_log_loss  top1  top3
                    lightgbm   0.26576 0.08113  0.86857 0.00874        1.33946  48.3% 83.7%
                     xgboost   0.26606 0.08112  0.86868 0.00755        1.34385  48.4% 83.2%
         logistic_regression   0.26814 0.08198  0.86627 0.00959        1.35273  47.7% 84.2%
logistic_regression+isotonic   0.26944 0.08258  0.86374 0.00674        1.35608  48.0% 83.9%
            xgboost+isotonic   0.27072 0.08173  0.86621 0.01078        1.37345  47.8% 83.3%
           lightgbm+isotonic   0.27314 0.08137  0.86420 0.00594        1.39823  47.9% 82.9%
            lightgbm+sigmoid   0.27856 0.08296  0.86857 0.03628        1.41928  48.3% 83.7%
             xgboost+sigmoid   0.27924 0.08297  0.86868 0.03575        1.42188  48.4% 83.2%
 logistic_regression+sigmoid   0.28424 0.08444  0.86627 0.03953        1.45032  47.7% 84.2%
```

Race log loss 1.339 against a uniform baseline of 2.079 — **+35.6% skill**.

⚠️ **These numbers come from synthetic data and mean nothing about real racing.**
The generator's market is 85% informed and its noise is mild, so 0.87 AUC is an
artefact of the simulation. Published work on real UK racing lands nearer
0.75–0.80. The numbers demonstrate that the pipeline works, not that the model does.

### Calibration: measured, not assumed

The specification calls calibration mandatory, and the pipeline implements both
Platt scaling and isotonic regression. The honest result on this data is that
**calibration made log loss worse for every model**:

| | raw | +isotonic | +sigmoid |
|---|---|---|---|
| log loss | **0.2658** | 0.2731 | 0.2786 |
| ECE | 0.0087 | **0.0059** | 0.0363 |

Isotonic improved reliability (ECE 0.0087 → 0.0059) while costing log loss.
Platt made both worse. The reason is that the models were *already* well
calibrated — ECE under 0.01 — because they are trained with log loss (a proper
scoring rule) and, deliberately, **without any class rebalancing**. Fitting a
correction on 2,816 rows containing ~350 winners then adds more noise than it
removes, and Platt's rigid two-parameter sigmoid actively distorts a curve that
was already straight.

This is why `CalibrationComparison.improved` requires ECE to fall *and* log loss
not to rise. Calibration stays in the pipeline because real gradient-boosted
models on real racing data are usually over-confident — but it is evaluated
every run, never assumed.

The `--no-market` control is the other honest check:

| | with market features | without |
|---|---|---|
| race log loss | 1.339 | 1.411 |
| skill vs uniform | 35.6% | 32.2% |

The non-market features carry real signal on their own. That matters, because a
model that mostly re-derives the bookmaker's price cannot beat it.

### Reproducibility

Every registered version records what it is, what it was built from and how good
it was:

```
models/
  registry.json          index + champion pointer
  lightgbm/v1/
    estimator.joblib     the fitted model
    preprocessor.joblib  imputation, scaling, encoding
    feature_schema.json  column contract + fingerprint
    calibrator.joblib
    model.json           params, seed, feature version, library versions
    metrics.json
```

The feature schema is hashed, and loading refuses to proceed on a mismatch — a
renamed or reordered column is the most common way a working model starts
producing quiet nonsense. Promotion is a pointer change, so rolling back is
instant.

### Example output

```
Race rac_s006205    2026-06-30
Model: lightgbm
----------------------------------------------------
  rank  probability  horse
     1       51.7%  Synthetic Horse 496
     2       19.0%  Synthetic Horse 728
     3       14.1%  Synthetic Horse 53
     ...
----------------------------------------------------
   sum      100.0%
```

```json
{"horse_id": "hrs_s0496", "probability": 0.5167, "rank": 1}
```

---

## Value betting

Phase 5 answers the only question that matters commercially: *if we had followed
these signals, would we have made money?*

```bash
make signals                  # walk-forward predictions (expensive, cached once)
make strategy-compare         # every strategy over identical predictions
make market-dependency        # does the model add anything the market lacks?
make sensitivity              # how fast does the edge die under costs?
```

### Read this before any number below

The Racing API subscription is still inactive, so every result here comes from
**synthetic data whose market was built with 15% noise by construction**
(`market_efficiency=0.85`). An exploitable edge therefore *exists by design*. A
profitable backtest demonstrates that the engine finds an edge that is really
there — a positive control. **It says nothing whatsoever about real racing**,
where the market is far more efficient and where these numbers would very likely
collapse.

### Two separations that stop the engine flattering itself

**Consensus price for probability, best price for the bet.** The best price
across many bookmakers is not the market's opinion — it is the opinion of
whichever book is slowest. Its implied probabilities often sum to *under* 1, and
dividing by that "overround" manufactures edge from nothing. Probability comes
from the consensus (mean) price; the bet is struck at the best.

**Predictions generated once, replayed through every strategy.** Re-running
walk-forward training per strategy would confound strategy differences with
differences between independently retrained models.

### Walk-forward, not a single split

Phase 4's one-shot split is the right way to measure a *model*. A *strategy*
would have been retrained as data arrived, so the backtest retrains annually and
predicts only forward:

```
train 2018-01-01→2019-12-17  predict 2020   (732 races)
train 2018-01-01→2020-12-17  predict 2021   (730 races)
...
train 2018-01-01→2025-12-17  predict 2026H1 (362 races)
```

### Strategy comparison

4,746 races, flat £10, 2% slippage, no commission:

```
    strategy  bets  strike  avg_odds     roi   profit  profit_factor  max_dd  t_stat
      B_ev10  3732   40.0%      3.57  +19.5%  7293.03          1.326   1.4%    7.21
  C_top_pick  4147   45.2%      3.01  +18.9%  7841.44          1.345   1.8%    8.56
 D_ev5_tight  4036   40.5%      3.50  +18.7%  7559.35          1.315   2.1%    7.37
       A_ev5  4040   40.4%      3.54  +18.6%  7522.98          1.312   2.1%    7.24
E_unfiltered  7792   34.8%      6.68   +8.0%  6195.16          1.122   4.4%    4.16
```

The unfiltered control earns its place: dropping the guards doubles the bet count,
pushes average odds from 3.5 to 6.7, and halves ROI. That is the longshot trap
the filters exist to prevent — `EV = p·odds − 1` multiplies model error by the
price, so an unguarded EV rule converts calibration noise in the tail into
confident bets on 40/1 shots.

### Market dependency — the decisive experiment

```
         strategy  bets  strike  avg_odds     roi   profit  t_stat  verdict
     1_full_model  4040   40.4%      3.54  +18.6%  7522.98    7.24  positive expectancy
    3_market_only     0       —         —       —        —       —  no bets placed
2_no_market_model  4162   22.6%      6.28   −5.6% −1896.28   −1.75  negative expectancy
```

Three things to take from this table:

**The negative control passes.** Betting the market's own margin-free probability
finds *zero* opportunities. It must — its EV is `1/overround − 1`, always
negative. Had it profited, the framework would be broken.

**The market-free model loses money.** Strip `mkt_*` and `market_*` features and
the model bets longer prices (6.28 vs 3.54), strikes at 22.6%, and returns −5.6%.

**So the edge is not independent alpha.** The full model works by *combining*
market information with fundamentals — it denoises the price rather than
replacing it. That is a legitimate and common way to beat a market, but it is a
much weaker claim than "our model handicaps better than the bookmakers", and it
depends entirely on the market being noisy enough to denoise. This synthetic
market is 15% noise. A real one is not.

### How fragile is it?

```
slippage commission  bets     roi  profit  t_stat
      0%         0%  4040  +21.0%  8501.0    8.02
      2%         0%  4040  +18.6%  7523.0    7.24
      5%         0%  4040  +15.0%  6056.0    6.02
      2%         5%  4040  +14.7%  5942.0    5.94
      5%         5%  4040  +11.3%  4548.0    4.69
     10%         5%  4040   +5.5%  2225.0    2.42
```

A 21% gross edge becomes 5.5% under pessimistic-but-plausible execution. On a
real market with a fraction of this edge, the same costs would erase it entirely.
Nothing here models a bookmaker restricting a winning account, which is what
actually ends most profitable betting operations.

### Where the money comes from

```
    band        bets   strike      profit      roi
    1-2          517   68.1%         +941   +18.2%
    2-3         1286   51.7%       +3,449   +26.8%
    3-5         1589   32.0%       +3,409   +21.5%
    5-8          582   16.7%         -408    -7.0%
    8-13          61   11.5%          +22    +3.6%
```

Profit is concentrated at short prices and turns negative above 5.0 — consistent
with the model being reliable where it has data and unreliable in the tail. The
calibration table on placed bets confirms it: the ratio of realised strike rate
to mean predicted probability is 0.84 in the lowest-probability band and ~1.0
everywhere else.

Monthly: 56 of 78 months profitable, worst −£128, best +£366. Maximum drawdown
2.05% of peak (flat staking on a growing bankroll).

### Verdict

**On synthetic data: yes, positive expected value, and statistically clear
(t = 7.2 over 4,040 bets).** The engine detects an edge that was deliberately
placed there, rejects the market-only control, and degrades sensibly under costs.

**On real racing: unproven, and the market-dependency result is a warning.** The
edge lives in market denoising, so it scales with how inefficient the market is.
The honest next step is not to tune these strategies — it is to get real data.

---

## How the product works

```
Racing API → today's racecards → FeaturePipeline → production model
           → calibrated probabilities → consensus price → EV
           → betting rules → BET / NO_BET → predictions table → dashboard
```

Every stage is **the same code the backtest used**. `FeaturePipeline`,
`RacePredictor`, `build_market_frame` and `compute_signal_frame` are imported by
the live service rather than reimplemented, so a signal generated this morning
comes from the identical path that generated the historical ones.

That claim is proved rather than asserted:

```bash
$ make replay day=2026-06-15
Backtest / live consistency
  Status            CONSISTENT
  Runners checked   16
  Largest gap       0.00e+00
```

Nobody sets out to build two models. It happens because the live service needs
one small thing the backtest did not — a column renamed, a NaN filled — and each
change is individually reasonable. Six of them later the live model is scoring a
different feature matrix, the backtest still reports 6% ROI, and nothing has
failed. `make replay` is what notices.

### The endpoints

| Endpoint | What it returns |
|---|---|
| `GET /predictions/today` | what the morning job stored |
| `GET /predictions/{race_id}` | one race |
| `POST /predict/today` | re-score against *current* prices |
| `GET /model/status` | which model is live, and the rules in force |
| `GET /performance` | the settled record |

`GET` reads what was true at 08:00; `POST` re-scores now. Prices move, so a
caller who needs the live position asks for it explicitly rather than getting it
by accident from a GET.

### The performance page counts only settled bets

An unsettled prediction is not a pending win, and including it would flatter
every number on the page during a losing week. ROI is shown with its
t-statistic, because 40 bets at +12% and 4,000 bets at +12% are not the same
claim — under 200 settled bets the page says so in words.

---

## Real data validation

Phase 6 answers one question: *does this make money on real UK racing?* The
apparatus is complete and tested. **The answer is not available**, because the
Racing API subscription on these credentials is inactive — see
[PHASE6_STATUS.md](PHASE6_STATUS.md) for the evidence that this is a billing
state rather than a bug.

Four commands, in order. Each refuses to run until the one before it has really
succeeded:

```bash
make api-status        # auth, subscription, endpoints, and the history range served
make phase6-preflight  # READY FOR VALIDATION, or BLOCKED with the one reason why
make phase6-run        # the whole workflow — this is the one button
```

`make phase6-run` executes seven steps in order — download, audit, features,
train, predict, backtest, report — checkpointing after each one. It is safe to
interrupt: a re-run picks up at the step that stopped rather than repeating an
eight-hour download or a forty-minute retrain. `make phase6-steps` shows how far
the last attempt got, with per-step runtimes.

Four documents come out of it:

| File | What it is |
|---|---|
| `REAL_DATA_AUDIT.md` | coverage, quality and timestamp consistency of the data |
| `Phase6_Final_Report.md` | the full research report, eight sections |
| `Horse_Racing_Quant_Strategy_Report.pdf` | the client-facing report, nine sections |
| `PRODUCTION_READINESS.md` | API / Data / Model / Backtest / Risk — PASS or FAIL |

### `make api-status` — what the plan can actually do

Authentication and subscription are reported **separately**, because they are
different problems with different owners. A rejected password is yours to fix; a
lapsed plan is billing's. The API distinguishes them and so does this:

```
Racing API status
-----------------
  Username           m7tL********************
  Authentication     PASS
  Subscription       INACTIVE

  Available endpoints
    Racecards    SKIPPED  Racing API subscription is inactive
    Results      SKIPPED  Racing API subscription is inactive
    Odds         SKIPPED  Racing API subscription is inactive
    Historical   SKIPPED  Racing API subscription is inactive

  Ready for backfill NO
```

`SKIPPED` rather than `NO` is deliberate: once the subscription has refused us
once, probing four more endpoints tells us nothing and hammers an API that has
already said no.

When the plan *is* live it also reports the window actually served:

```
  Data range
    Earliest result   2014-01-02
    Latest result     2026-08-09
    Span              12.6 years

  Coverage
    Best racecard tier   pro
    Protocol needs from  2018-01-01
    Enough for protocol  YES
```

The earliest date is found by bisecting the year — about five calls across a
twenty-five year range, not twenty-five. And *enough for protocol* is not the
same question as *is there a lot of history*: a plan serving 2021 onward has five
years and still cannot run a study that trains from 2018. Learning that from one
probe is considerably cheaper than learning it from eighty empty months of
backfill.

### The plan was frozen before the data existed

Phase 6's instruction was *do not optimize after seeing results*. A promise is
not a mechanism, so the analysis plan lives in
[`backend/research/protocol.py`](backend/research/protocol.py) as a
content-hashed dataclass written before a single real race was downloaded:

```
$ make protocol
protocol     phase6-v1
registered   2026-08-10
fingerprint  76e18ddb6f11f6ad
```

Fixed in advance: the splits (train 2018–2022, validate 2023–2024, test
2025–2026, 14-day embargo); **one** primary hypothesis (LightGBM · `A_ev5` ·
flat stakes · 2% slippage · 2% commission), named so it cannot be picked later
from whichever cell of the results grid looks best; and the decision rule —
≥200 bets **and** t ≥ 2.0 **and** profit in ≥50% of years.

Twelve secondary tests are pre-declared, which drags the Bonferroni threshold
from 2.0 to **2.89**. A segment that scrapes t = 2.3 does not count as an edge,
and cannot be argued into one after the fact.

`ValidationProtocol.verdict()` maps the numbers onto one of four conclusions
mechanically — `positive_edge_confirmed`, `no_edge_found`,
`edge_only_in_specific_segments`, `insufficient_data_to_conclude`. No prose in
the report can move it. Edit the protocol after seeing results and the
fingerprint changes, which fails the run.

### The gates that matter most

`make validate` runs six named gates in order and **stops at the first failure**,
reporting which one stopped it. "Validation failed" sends someone reading
tracebacks; "STOPPED at 'data source'" sends them to fix the right thing.

| Gate | Stops the run when |
|---|---|
| protocol integrity | the frozen plan's fingerprint has changed |
| data source | the database holds synthetic or mixed races |
| sufficient data | fewer than 5 full racing years |
| odds coverage | under 60% of runners priced, or under 95% settled |
| timestamp integrity | any quote is recorded after the off |
| no leakage | a single feature separates winners almost perfectly |

The source gate is the most important guard in the project. A validation report
generated from generated data would be indistinguishable from a real one at a
glance, and would be the single most dangerous artefact this repository could
produce.

The leakage gate runs last because it needs a built feature matrix, and it
checks by *behaviour* rather than by name — a denylist only catches leaks
somebody already imagined. It found a real one in Phase 4 and, on its first run
here, a false positive: `finishing_position` is carried in the prediction frame
so bets can be settled, and separates winners perfectly by construction. The fix
was to exclude exactly `LABEL_COLUMNS`, the authoritative list that lives next to
the query producing those columns — because a hand-written copy of that list is
precisely how the Phase 4 leak happened.

The machinery has still been exercised end to end — `make validate-harness`
overrides the gates on synthetic data and stamps the output **"Results are a
harness check, not findings."**

### Does the model know anything the price does not?

Comparing log losses cannot answer that: a model fed market features always
scores near the market, and "slightly better" is usually noise. The instrument
in [`backend/ml/baselines.py`](backend/ml/baselines.py) is a forecast
**encompassing** regression:

```
logit P(win) ~ logit(p_market) + logit(p_model)
```

The price is already in the equation, so the model's coefficient measures what
it adds *conditional on* the market being known. `b_model ≈ 0` means the market
encompasses the model — whatever it knows, the price knew first. On real UK
racing that is the expected result, and `no_edge_found` is a perfectly
respectable Phase 6 outcome. Being able to report it without embarrassment is
exactly why the protocol was written first.

The model is also scored against random, uniform, favourite and market-consensus
baselines, then segmented by year, code, handicap status, price band and field
size — because an edge that exists in one year and one price band is an artefact,
not a strategy.

### Every result is reproducible, or it does not count

Each run is recorded in `research_runs` with three hashes, because
reproducibility needs all three: the **protocol** hash says which hypothesis was
tested, the **dataset** hash says what it was tested on, and the **feature
version** says how those rows became inputs. `make experiments` lists them and
flags any run missing one. A verdict carrying all three can be re-derived; a
verdict carrying none of them is an anecdote.

Changing the protocol invalidates the entire execution checkpoint rather than
resuming across it. A half-old, half-new run is not a study of anything — it is
two studies averaged together, and nobody could say which.

### The reports

`Phase6_Final_Report.md`, in eight sections: dataset, model performance,
probability calibration, betting strategy performance, year-by-year results,
segment analysis, statistical significance, and the final verdict — exactly one
of `EDGE_CONFIRMED`, `NO_EDGE_FOUND`, `SEGMENT_EDGE` or `INCONCLUSIVE`.

Calibration gets its own section because ranking well is not enough: stakes are
computed from the probability, not the rank, so a 20% shout that wins 12% of the
time turns a real edge into a real loss.

The client PDF adds risk and drawdown sections and carries one hard rule: **it
cannot describe the strategy as profitable unless the frozen decision rule says
`EDGE_CONFIRMED`.** The executive summary selects its wording from the verdict
enum, never from a number — a glowing ROI with a t-statistic of 1.1 produces the
same "this does not support staking money on it" paragraph as a loss. That is
asserted by reading the generated PDF's text back and checking the encouraging
phrases are absent.

`PRODUCTION_READINESS.md` answers a different question again: is each component
*fit to be relied on*. A full row of PASSes means the pipeline is sound, the data
is clean, the probabilities are calibrated, the backtest was honest and the risk
controls hold — it does **not** mean there is an edge. Conflating those two is
how a green dashboard ends up funding a losing strategy, so the checklist prints
the verdict verbatim and draws no conclusion of its own from it.

### The backfill is auditable, not just resumable

`make history` checkpoints to disk after every month so it can resume, and
records every invocation in the `backfill_runs` table so it can be audited:
window, status, records written, records rejected, months completed and failed,
and why it stopped. `make history-runs` prints the trail.

The two are not redundant. The checkpoint answers "which month do I do next?";
the table answers "was this month ever successfully imported?" — which is the
first question worth asking when the audit later reports a thin year. A run that
finishes with a failed month is recorded as `partial`, never `completed`, because
a partial import reporting success is how a dataset ends up with a silent hole
in it.

---

## Configuration

All configuration is centralised in [`backend/utils/config.py`](backend/utils/config.py)
and loaded from the environment. **No module outside that file may read
`os.environ`** — there is a test (`tests/unit/test_structure.py`) that fails the
build if one does.

Credentials are typed as `SecretStr`, so they are masked in every `repr()`,
log line, and API response. `settings.safe_database_url` is the only form of the
connection string that is ever logged.

Copy `.env.example` to `.env` and fill it in; `.env` is git-ignored.

| Group | Key settings |
|-------|--------------|
| App | `ENVIRONMENT`, `DEBUG`, `API_PORT`, `API_V1_PREFIX` |
| Logging | `LOG_LEVEL`, `LOG_JSON`, `LOG_DIR` |
| Database | `POSTGRES_*` or a single `DATABASE_URL` |
| Racing API | `RACING_API_USERNAME`, `RACING_API_PASSWORD`, `RACING_API_KEY` |
| Strategy | `MIN_EXPECTED_VALUE`, `KELLY_FRACTION`, `MAX_STAKE_FRACTION`, `STARTING_BANKROLL` |

`ENVIRONMENT=staging|prod` activates guardrails: the app refuses to start with the
placeholder `SECRET_KEY` or with `DEBUG=true`, and the OpenAPI docs are disabled.

---

## Logging

Three channels, three rotating files, plus a console stream and an `error.log`
that mirrors everything at `ERROR` and above:

```python
from backend.utils.logging import get_logger

log = get_logger(__name__, channel="pipeline")
log.info("ingested racecards", extra={"race_date": "2026-08-10", "races": 42})
```

```
logs/api.log        HTTP requests, Racing API client
logs/pipeline.log   ETL, ingestion, feature builds
logs/model.log      training, inference, backtests
logs/error.log      everything ERROR and above, all channels
```

Every request is stamped with an `X-Request-ID` (accepted from the caller or
generated). That id rides a `ContextVar` onto **every** log line produced while
handling the request, across all three channels — so one grep reconstructs an
entire incident. Set `LOG_JSON=true` in staging/production for structured output.

---

## Database

Synchronous SQLAlchemy 2.0, deliberately: the ETL, feature, training and
backtesting layers are pandas/NumPy code that is synchronous anyway, and FastAPI
runs sync endpoints in a threadpool. One engine, one mental model.

```python
from backend.database import session_scope

with session_scope() as session:      # commits on success, rolls back on error
    session.add(race)
```

The engine is created lazily, so importing any module never requires a live
database — tests, notebooks and Alembic all work offline.

Every constraint and index is named by the convention in
[`backend/database/base.py`](backend/database/base.py). This is not cosmetic:
without it, Alembic generates constraints that cannot be dropped, and migrations
become one-way.

```bash
make migration m="add races table"   # autogenerate
make migrate                         # upgrade head
make downgrade                       # roll back one
```

---

## Testing

```bash
make test              # everything, with coverage
make test-unit         # no database required
make test-integration  # requires `make db-up`
make check             # lint + typecheck + test (what CI runs)
```

Tests run against a temporary SQLite file, a throwaway log directory and dummy
credentials. `HQP_ENV_FILE` is repointed at a nonexistent path in `conftest.py`
so a developer's real `.env` can never leak into a test run.

Integration tests skip themselves when no database answers, so `make test` stays
green on a laptop with nothing running. They connect via `TEST_POSTGRES_*`, kept
separate from the app's own `POSTGRES_*` so a test run can never touch the real
database — and so a container on a non-default port still works:

```bash
POSTGRES_PORT=5434 make db-up
TEST_POSTGRES_PORT=5434 make test
```

Beyond the usual unit tests, `tests/unit/test_structure.py` enforces the
architecture itself: required packages import, required files exist, `.env` is
git-ignored, the template ships no real secrets, and nothing bypasses `Settings`.

---

## Development commands

```bash
make help              # list every target
make format            # ruff format + autofix
make lint              # ruff check
make typecheck         # mypy
make lock              # freeze resolved versions into requirements.lock.txt
make db-shell          # psql into the container
make clean             # drop caches and build artefacts
```

---

## Responsible use

This is a research platform. Betting markets are efficient enough that most
apparent edges are overfitting, stale prices, or unmodelled costs. The backtest
engine (Phase 8) is built to be *pessimistic* — walk-forward only, no look-ahead,
commission and slippage included — because the expensive mistake is believing a
backtest that flatters you. Never stake money you cannot afford to lose.
