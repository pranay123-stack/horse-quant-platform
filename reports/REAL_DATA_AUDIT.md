# Real data audit

*Generated 2026-08-10T01:36:46.228680+00:00*

**Dataset source: `synthetic`**
**Modelling readiness: NOT READY**

## Blocking issues

- this dataset is synthetic — it cannot validate anything about real markets

> A study run on this dataset cannot support a conclusion about real markets.

## Totals

| Entity | Count |
|---|---:|
| Races | 6,206 |
| Runners (declarations) | 49,648 |
| Results | 49,648 |
| Horses | 1,200 |
| Jockeys | 90 |
| Trainers | 70 |
| Odds quotes | 360,256 |

## Span

- **2018-01-01 → 2026-06-30**
- 3,103 distinct racing days across 9 year(s)

## Coverage

| Check | Value | Threshold |
|---|---:|---:|
| Runners with a price | 90.7% | 60% |
| Races with results | 100.0% | 95% |
| Races missing results | 0 | — |
| Races missing runners | 0 | — |
| Runners missing odds | 4,616 | — |

## Quality

| Check | Count | Why it matters |
|---|---:|---|
| Duplicate course/date/time slots | 0 | the same race counted twice |
| Prices at or below evens | 0 | an impossible decimal price |
| Finished races with no winner | 0 | an unlabelled row in the training set |
| Horses with a single career run | 0 | no form history to build features from |

### Timestamp consistency

Every leakage guarantee in this project rests on knowing when a price was taken relative to the off. These three checks are what make that knowable.

| Check | Count | Consequence |
|---|---:|---|
| Quotes recorded after the off | 0 | a post-race price is not bettable — **blocking** |
| Races with no off time | 0 | their quotes cannot be proven pre-race |
| Off time on a different day than race_date | 0 | shifts the point-in-time feature cutoff |

## Coverage by year

| Year | Races | Days | Runners | Result coverage | Odds coverage |
|---|---:|---:|---:|---:|---:|
| 2018 | 730 | 365 | 5,840 | 100.0% | 91.0% |
| 2019 | 730 | 365 | 5,840 | 100.0% | 89.3% |
| 2020 | 732 | 366 | 5,856 | 100.0% | 91.3% |
| 2021 | 730 | 365 | 5,840 | 100.0% | 90.7% |
| 2022 | 730 | 365 | 5,840 | 100.0% | 92.1% |
| 2023 | 730 | 365 | 5,840 | 100.0% | 90.3% |
| 2024 | 732 | 366 | 5,856 | 100.0% | 90.3% |
| 2025 | 730 | 365 | 5,840 | 100.0% | 89.5% |
| 2026 | 362 | 181 | 2,896 | 100.0% | 93.4% |

## Race types

| Type | Races |
|---|---:|
| Chase | 3,103 |
| Flat | 3,103 |
