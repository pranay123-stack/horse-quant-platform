# Production readiness

*Generated 2026-08-10T01:36:48.177635+00:00*
*Protocol `phase6-v1` · fingerprint `76e18ddb6f11f6ad`*

| Component | Status |
|---|---|
| API | **FAIL** |
| Data | **FAIL** |
| Model | **PASS** |
| Backtest | **PASS** |
| Risk | **PASS** |

**Overall: NOT READY**

> These checks answer *is each component fit to be relied on* — not *is the
> strategy profitable*. A row of PASSes means the pipeline is sound, the data
> is clean, the probabilities are calibrated, the backtest was run under
> realistic conditions and the risk controls hold. Whether an edge exists is
> a different question, with exactly one answer, below.

## Detail

### API — FAIL

Racing API subscription is inactive (Subscription inactive)

### Data — FAIL

this dataset is synthetic — it cannot validate anything about real markets

### Model — PASS

calibrated and better than an uninformed forecast

- calibration error 0.0091
- 8,736 rows scored
- beats uniform (1.3459 vs 2.0794)

### Backtest — PASS

6 strategies over 940 primary bets, walk-forward, 2% slippage and 2% commission

- no look-ahead: starting price excluded
- 2 years segmented

### Risk — PASS

maximum drawdown 2.2%, longest losing streak 10 bets

- bankroll 10,000 → 11,534
- average stake 10.00

## What must be fixed

- **API**: Racing API subscription is inactive (Subscription inactive)
- **Data**: this dataset is synthetic — it cannot validate anything about real markets

## Profitability verdict

### `EDGE_CONFIRMED`

Positive edge confirmed: +16.32% ROI over 940 bets (t = 3.11)

*Produced by the pre-registered decision function, which was content-hashed before any data existed. Nothing in this checklist can move it, and a full row of PASSes above does not imply it.*
