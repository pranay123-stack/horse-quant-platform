# Phase 6 — real data validation: status

**Status: built, tested, and blocked on one external dependency.**

Phase 6 asks a single question — *does this model make money on real UK racing?*
Every piece of machinery needed to answer it exists and is tested. The answer
itself cannot be produced, because The Racing API subscription attached to these
credentials is inactive.

---

## The blocker

```
$ make api-status
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

  Best racecard tier -
  History to 2018-01-01  NO

  Ready for backfill NO
```

Authentication and subscription are reported separately because they are
different problems with different owners. `SKIPPED` rather than `NO` is also
deliberate: once the plan has refused us once, probing four more endpoints tells
us nothing and hammers an API that has already said no.

Verified again on 2026-08-10. The distinction matters and is worth being precise
about:

| Probe | Response | What it proves |
|---|---|---|
| Correct credentials, any endpoint | `{"detail":"Subscription inactive"}` | The account exists and is recognised |
| Deliberately wrong password | `{"detail":"Incorrect password"}` | Authentication is working; the credentials are right |
| `/openapi.json` | Full schema, 200 OK | The host is reachable; nothing is wrong with the network |

So this is not a bug, a credential problem, or a connectivity problem. It is a
billing state. Every endpoint — including the free tier — returns the same
refusal until a plan is enabled at <https://www.theracingapi.com>.

**Nothing in this repository can work around it, and nothing tries to.** Phase 6
is about real data. Substituting anything else would defeat the phase.

---

## What was built anyway

Everything that does not require the data to exist. The list below is complete
and each item is covered by tests.

| Phase | Component | Module | State |
|---|---|---|---|
| 6.1 | API capability probing | `backend/services/racing_api/capabilities.py` | ✅ auth vs subscription separated |
| 6.2 | Resumable + audited backfill | `backend/data_pipeline/backfill.py`, `backfill_runs` | ✅ checkpoint + DB audit trail |
| 6.3 | Data audit → `REAL_DATA_AUDIT.md` | `backend/research/audit.py` | ✅ incl. timestamp consistency |
| 6.4 | Pre-flight gates + validation runner | `backend/research/preflight.py` | ✅ 6 gates, stop at first failure |
| 6.5 | Final report → `Phase6_Final_Report.md` | `backend/research/validation_report.py` | ✅ 8 sections, verdict from the frozen rule |
| 6.6 | Tests | `tests/unit/test_phase6_*.py` | ✅ |
| 6.7.1 | Data range + protocol coverage in `api-status` | `capabilities.py` | ✅ bisects the year |
| 6.7.2 | `make phase6-preflight` | `preflight.py` (`full_preflight`) | ✅ 8 gates, API first |
| 6.7.3 | `make phase6-run` — 7 resumable steps | `research/execution.py` | ✅ resume halves the runtime |
| 6.7.4 | `research_runs` experiment ledger | `models/operations.py` | ✅ 3 hashes per run |
| 6.7.5 | Client PDF | `research/client_report.py` | ✅ cannot claim profit without the verdict |
| 6.7.6 | `PRODUCTION_READINESS.md` | `research/readiness.py` | ✅ API/Data/Model/Backtest/Risk |
| — | Point-in-time rebuild | *(Phase 3 pipeline, unchanged)* | ✅ reused deliberately — see below |
| — | Baselines + encompassing test | `backend/ml/baselines.py` | ✅ |
| — | Robustness segmentation | `backend/strategy/robustness.py` | ✅ |
| — | **Pre-registered protocol** | `backend/research/protocol.py` | ✅ frozen, fingerprinted |

The instruction *do not reuse synthetic feature tables* refers to the feature
**tables**, not the feature **code**. The `data source` gate enforces exactly
that: it refuses to run on any database containing synthetic races, so a real
run necessarily rebuilds every feature from real rows.

---

## The pre-registered protocol

Phase 6's instruction was *do not optimize after seeing results*. A promise is
not a mechanism, so the analysis plan was instead written into code **before any
real data existed** and content-hashed:

```
protocol  phase6-v1
registered 2026-08-10
fingerprint 76e18ddb6f11f6ad
```

Frozen before the fact:

- **Splits** — train 2018-01-01→2022-12-31, validate 2023-01-01→2024-12-31,
  test 2025-01-01→2026-12-31, 14-day embargo.
- **Primary hypothesis** — LightGBM, strategy `A_ev5`, flat staking, 2% slippage,
  2% commission. *One* hypothesis, named in advance, so it cannot be chosen later
  from whichever cell of the results table looks best.
- **Decision rule** — ≥200 bets, t ≥ 2.0, and profit in ≥50% of years. All three,
  or no edge is claimed.
- **Multiplicity** — 12 secondary tests are pre-declared, so the Bonferroni
  threshold is 2.0 → **2.89**. A segment that scrapes past t=2.3 does not count.

`ValidationProtocol.verdict()` maps results onto one of four conclusions
mechanically. No prose in the report can move it. If the protocol is edited after
data arrives, the fingerprint changes and `assert_protocol_unchanged` fails the
run — that is the enforcement.

Run `make protocol` to print the plan.

---

## What happens the moment a subscription is enabled

Four commands, no code changes:

```bash
make api-status        # confirms ACTIVE, and that history reaches 2018
make phase6-preflight  # READY FOR VALIDATION, or BLOCKED with the reason
make phase6-run        # ← the one button
```

`make phase6-run` executes seven steps — download, audit, features, train,
predict, backtest, report — checkpointing after each. Interrupting it is safe:
a re-run resumes at the step that stopped. Measured on the harness, resuming
takes 67s against 134s for a full restart, because the walk-forward retrain is
genuinely skipped rather than merely reported as skipped.

Four documents come out:

| File | What it is |
|---|---|
| `REAL_DATA_AUDIT.md` | coverage, quality, timestamp consistency |
| `Phase6_Final_Report.md` | the research report, eight sections |
| `Horse_Racing_Quant_Strategy_Report.pdf` | the client report, nine sections |
| `PRODUCTION_READINESS.md` | API / Data / Model / Backtest / Risk |

`make history` checkpoints after every month, so it survives interruption and
resumes rather than restarting, and records each invocation in `backfill_runs`
so it can be audited afterwards (`make history-runs`). A run that finishes with a
failed month is recorded as `partial`, never `completed` — a partial import
reporting success is how a dataset ends up with a silent hole in it.

Eight gates run in order, stopping at the first failure and naming it. API
gates come first because they are the cheapest and the most likely to fail —
there is no point auditing a database that could never have been filled:

| Gate | Stops the run when |
|---|---|
| api subscription | the plan is not live, or the credentials are wrong |
| api endpoints | results or history are not included in the plan |
| history range | the served window does not reach the protocol's start |
| protocol integrity | the frozen plan's fingerprint has changed |
| data source | the database holds synthetic or mixed races |
| sufficient data | fewer than 5 full racing years |
| odds coverage | under 60% priced, or under 95% settled |
| timestamp integrity | any quote is recorded after the off |
| no leakage | one feature separates winners almost perfectly |

Every run is recorded in `research_runs` with a protocol hash, a dataset hash
and a feature version. All three are needed to re-derive a result, and
`make experiments` flags any run missing one.

The verdict is one of `EDGE_CONFIRMED`, `NO_EDGE_FOUND`, `SEGMENT_EDGE` or
`INCONCLUSIVE`, produced by the frozen decision function. The labels are display
strings; the rule that selects between them has not been touched.

The harness itself has been exercised end to end — `make validate-harness` runs
the whole chain on synthetic data and produces a report stamped **"Results are a
harness check, not findings."** That flag is not decorative; it is the difference
between a validation and a very convincing-looking piece of fiction.

---

## The honest expectation

Phase 5 established the finding that frames all of this: with market features
removed, the model returned **−5.6% ROI**. Its apparent edge came from denoising
the market price, not from independent information — and it scaled with how noisy
that market was. The synthetic market is 15% noise by construction. A real
bookmaker market is not.

The encompassing test in `backend/ml/baselines.py` is the instrument that will
settle it. It fits

```
logit P(win) ~ logit(p_market) + logit(p_model)
```

and reads the coefficient on the model *conditional on the price already being
known*. On real UK racing, `b_model ≈ 0` — the market encompasses the model — is
the expected result, and `no_edge_found` is a perfectly good outcome for Phase 6.
It is the outcome the protocol was designed to be able to report without
embarrassment, which is precisely why the protocol was written first.

The value of this phase is not that it will find an edge. It is that if it
reports one, the report will be worth believing.
