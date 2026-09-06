# Phase 6 — final validation report

*Generated 2026-08-10T01:36:48.177635+00:00*
*Protocol `phase6-v1` · fingerprint `76e18ddb6f11f6ad` · registered 2026-08-10*

> ⚠️ RUN ON A DATASET THAT FAILED A PRE-FLIGHT GATE — (data source: the database contains only synthetic races — generated data cannot validate anything about real markets). Results are a harness check, not findings.

## Verdict: `EDGE_CONFIRMED`

**Positive edge confirmed: +16.32% ROI over 940 bets (t = 3.11)**

- primary ROI +16.32% over 940 bets
- primary t = +3.11 (threshold 2.0)
- profitable years 100% of 2 (threshold 50%)
- model beats the market baseline on race log loss (-0.1336)

*Produced by the pre-registered decision function, which was fixed before any data existed and is applied mechanically. Nothing in this report can move it.*

## 0. Pre-flight gates

```
Pre-flight gates
----------------
  [PASS] protocol integrity     fingerprint 76e18ddb6f11f6ad (not pinned by the caller)
  [STOP] data source            the database contains only synthetic races — generated data cannot validate anything about real markets
  [PASS] no leakage             89 features, none suspicious

  STOPPED at 'data source'
```

## 1. Dataset

- Source: **synthetic**
- Span: **2018-01-01 → 2026-06-30** (9 years, 3,103 racing days)
- 6,206 races · 49,648 runners · 1,200 horses
- Odds coverage 90.7% · result coverage 100.0%
- Out-of-sample rows scored: 8,736
- Readiness: **NOT READY**

### Splits (pre-registered)

| Split | From | To |
|---|---|---|
| Train | 2018-01-01 | 2022-12-31 |
| Validate | 2023-01-01 | 2024-12-31 |
| Test (out of sample) | 2025-01-01 | 2026-12-31 |

Embargo between splits: **14 days**.

  - blocking: this dataset is synthetic — it cannot validate anything about real markets

## 2. Model performance

Race-level log loss on the out-of-sample window, against every baseline. Lower is better. The `market` row is the only comparison that decides anything — beating `uniform` or `random` is table stakes.

| forecaster | races | race_log_loss | top1_hit_rate | top3_hit_rate | skill_vs_uniform |
|---|---|---|---|---|---|
| model | 1092 | 1.34594 | 0.4881 | 0.8407 | 0.3527 |
| market | 1092 | 1.47951 | 0.4579 | 0.793 | 0.2885 |
| uniform | 1092 | 2.07944 | 0.12 | 0.3626 | 0.0 |
| favourite | 1092 | 2.19257 | 0.4579 | 0.6227 | -0.0544 |
| random | 1092 | 2.42684 | 0.0943 | 0.3498 | -0.1671 |

## 3. Probability calibration

Ranking well is not enough: stakes are computed from the probability, so a 20% shout that wins 12% of the time turns a real edge into a real loss. Measured on the test window, with the calibrator fitted on an earlier slice.

**Expected calibration error: 0.0091**

```
  predicted  observed      n  |0.0                              1.0|
  ---------  --------  -----  +-----------------------------------+
      0.003     0.001    874  |X·································|
      0.007     0.003    874  |X·································|
      0.016     0.008    873  |X·································|
      0.029     0.026    874  |X·································|
      0.047     0.038    873  |·X································|
      0.076     0.056    874  |·Op·······························|
      0.117     0.116    873  |···X······························|
      0.182     0.183    874  |······X···························|
      0.277     0.277    873  |·········X························|
      0.497     0.541    874  |················p·O···············|
  p = mean predicted, O = observed frequency, X = they coincide
```

## 4. Betting strategy performance

Identical predictions replayed through every strategy, at 2% slippage and 2% commission.

| strategy | bets | strike_rate | avg_odds | roi | profit | profit_factor | max_dd | sharpe | t_stat | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| G_ev15/kelly | 742 | 0.3989 | 3.65 | 0.2122 | 63160.83 | 1.36 | 0.173 | 0.1261 | 3.43 | positive expectancy (t=3.43) |
| G_ev15/flat | 742 | 0.3989 | 3.65 | 0.2079 | 1542.59 | 1.346 | 0.0188 | 0.1261 | 3.43 | positive expectancy (t=3.43) |
| B_ev10/kelly | 880 | 0.3955 | 3.62 | 0.1856 | 63925.06 | 1.316 | 0.1644 | 0.1049 | 3.11 | positive expectancy (t=3.11) |
| A_ev5/kelly | 940 | 0.3968 | 3.6 | 0.1801 | 65068.66 | 1.308 | 0.1724 | 0.1016 | 3.11 | positive expectancy (t=3.11) |
| B_ev10/flat | 880 | 0.3955 | 3.62 | 0.1696 | 1492.58 | 1.281 | 0.0229 | 0.1049 | 3.11 | positive expectancy (t=3.11) |
| A_ev5/flat | 940 | 0.3968 | 3.6 | 0.1632 | 1533.65 | 1.27 | 0.0223 | 0.1016 | 3.11 | positive expectancy (t=3.11) |

### Primary hypothesis — A_ev5 / flat

The one strategy named in advance. Every other row above is a secondary test and carries the corrected significance bar.

```
Strategy: A_ev5/flat
========================================================================
  Period            2025-01-01 → 2026-06-30
  Races considered  1092
  Bets              940  (86.1% of races)
  Winners           373
  Strike rate       39.68%
  Average odds      3.60
  Average stake     10.00

  Staked            9,400.00
  Profit / loss     +1,533.65
  ROI               +16.32%
  Profit factor     1.270

  Bankroll          10,000 → 11,533.65
  Peak              11,602.95
  Max drawdown      253.24 (2.23%)
  Longest losing    10

  Sharpe (per bet)  +0.1016
  ROI std error     0.0524
  t-statistic       +3.11
  Model calibration 1.008  (strike rate / mean predicted)

  VERDICT           positive expectancy (t=3.11)

  Equity curve
  ▂▂▂▁▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▅▄▅▅▅▅▆▅▆▆▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇█  [9,880 … 11,456]

  By price band
    band        bets   strike     staked      profit      roi
    1-2          117   70.1%       1,170        +234   +20.0%
    2-3          268   52.6%       2,680        +747   +27.9%
    3-5          396   33.1%       3,960      +1,031   +26.0%
    5-8          146   12.3%       1,460        -439   -30.1%
    8-13          13    7.7%         130         -39   -30.3%

  Calibration on placed bets
    predicted   actual   ratio   bets
        21.2%    16.0%    0.75    188
        28.7%    28.2%    0.98    188
        36.9%    36.7%    0.99    188
        47.2%    53.2%    1.13    188
        62.9%    64.4%    1.02    188

  Why runners were not backed
    probability_too_low            3288
    odds_too_high                  1651
    ev_below_threshold             1434
    no_odds                         808
    max_bets_per_race               573
    odds_too_low                     42
```

## 5. Year-by-year results

An edge that lives in one year is an artefact. The protocol requires profit in at least 50% of years.

```
  segment                  bets   strike  avg odds       roi       t
  ------------------------------------------------------------------
  year=2025                 615   39.8%      3.75   +19.7%   +2.95
  year=2026                 325   39.4%      3.31    +9.9%   +1.18
```

## 6. Segment analysis

```
Consistency
  Years covered        2
  Profitable years     2 (100%)
  Worst / best year    +9.95% / +19.68%
  Reportable segments  9
  Positive segments    8 (89%)
  Pass at t>=2.0       6
  Pass at t>=2.89 (corrected)  4
```

### By code

```
  segment                  bets   strike  avg odds       roi       t
  ------------------------------------------------------------------
  code=flat                 469   40.3%      3.55   +18.0%   +2.46
  code=national_hunt        471   39.1%      3.65   +14.7%   +1.95
```

### Handicap versus non-handicap

```
  segment                  bets   strike  avg odds       roi       t
  ------------------------------------------------------------------
  handicap=non_handicap     940   39.7%      3.60   +16.3%   +3.11
```

### By price band

```
  segment                  bets   strike  avg odds       roi       t
  ------------------------------------------------------------------
  odds_band=1-2             117   70.1%      1.74   +20.0%   +2.70
  odds_band=10-1000           3    0.0%     11.10  -100.0%   +0.00  (thin)
  odds_band=2-5             664   41.0%      3.34   +26.8%   +4.29
  odds_band=5-10            156   12.2%      5.96   -28.7%   -1.84
```

### By field size

```
  segment                  bets   strike  avg odds       roi       t
  ------------------------------------------------------------------
  field_size_band=<=8       940   39.7%      3.60   +16.3%   +3.11
```

## 7. Statistical significance

| Quantity | Value | Required |
|---|---:|---:|
| Bets (primary) | 940 | 200 |
| ROI (primary) | +16.32% | > 0 |
| t-statistic (primary) | +3.11 | 2.00 |
| Profitable years | 100% | 50% |
| Secondary-test bar (Bonferroni, 12 tests) | 2.89 | — |

### Does the model know anything the price does not?

**model adds information beyond the price (b_model = +0.785, z = 6.02)**

Encompassing regression `logit(win) ~ logit(p_market) + logit(p_model)`. The market's forecast is already in the equation, so the model's coefficient measures what it adds *conditional on the price being known*.

| Term | Coefficient | z |
|---|---:|---:|
| market | +0.5088 | +2.74 |
| model | +0.7850 | +6.02 |

Log loss: market only 0.26389 → combined 0.26144 (n = 7,928)

## 8. Final verdict

# `EDGE_CONFIRMED`

Positive edge confirmed: +16.32% ROI over 940 bets (t = 3.11)

Reasoning, in the order the frozen rule evaluates it:

1. primary ROI +16.32% over 940 bets
2. primary t = +3.11 (threshold 2.0)
3. profitable years 100% of 2 (threshold 50%)
4. model beats the market baseline on race log loss (-0.1336)

### Limitations

- Backtests are an upper bound: no model of account restriction, and liquidity is a flat cap rather than a real book.
- Bets settle at the last captured pre-race price less slippage; real execution is worse and prices move against a value bettor specifically.
- Walk-forward retrains annually. More frequent retraining would change results in an unknown direction.
- The out-of-sample window is scored once. It is one draw from a noisy distribution, which is what the t-statistic is there to express.
