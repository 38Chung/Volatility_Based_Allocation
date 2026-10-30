# QF5210 Group Project: Volatility-Based Allocation with Factor Selection on STI Constituents

This repository contains the full code, input data and results for our study of whether
(1) volatility-based allocation can reduce risk without giving up much return, and
(2) factor-based stock selection can beat the Straits Times Index (STI),
using the STI constituents as they were at each point in time (including companies that were later delisted).
All returns are in SGD. The risk-free rate is set to 0 and cash earns nothing, so Sharpe ratio = annualised return / annualised volatility.

Code comments are in Chinese; this README and all output column names are in English.

## 1. Repository layout

```
.
├── README.md                 this file
├── requirements.txt          Python packages
├── config.py                 dates, default parameters, universe, sectors, currencies, paths
├── run_stage1.py             Stage 1: data → universe → 19 factors → preprocessing → weekly IC tests
├── run_stage2.py             Stage 2: rolling Lasso → 8 portfolios → volatility experiments → parameter freeze
│                             (data truncated at the freeze date 2026-09-30)
├── run_stage3.py             Stage 3: test period with frozen parameters, sensitivity tests, figures, submission CSV
├── run_freq_compare.py       Control experiment: weekly vs monthly rebalancing (does not affect parameters)
├── fetch_mas_fx.py           Downloads MAS daily exchange rates (already saved in data/)
├── src/
│   ├── data_loader.py        reading data, FX conversion, delisting, trading days, tradable / membership masks
│   ├── factors.py            19 daily factors
│   ├── preprocess.py         weekly calendar, open-price index, holding returns, cross-sectional preprocessing
│   ├── ic_test.py            Rank IC, Newey–West t, IC decay, factor correlation
│   ├── model.py              rolling Lasso / Elastic Net, IC screen, λ selection
│   ├── volatility.py         stock volatility estimators, portfolio GARCH forecast
│   ├── portfolio.py          stock selection, weights, caps, volatility-target exposure
│   ├── backtest.py           daily NAV, Tuesday-open execution, costs, suspensions, delisting
│   ├── metrics.py            performance metrics, CAPM, Brinson, VaR/ES
│   └── plots.py              figures
├── tests/                    pytest checks (look-ahead, calendar, universe, backtest, λ rule, ...)
├── data/                     input data (see Section 2)
└── outputs/                  results (see Section 7)
```

## 2. Input data (`data/`)

| File | Content |
|---|---|
| `All_Stock_STI_used.csv` | Daily Datastream data for the 51 candidate stocks, 2010-01-01 to 2026-10-29, long format (`date`, `ticker` + the 18 fields used: RZ, P, UP, PO, MV, VO, UVO, VA, NOSH, UDD, EPS1FD12, MTBV, WC03501, PE, EPS, PH, PL, DY). Raw values in the quoting currency, exactly as downloaded. |
| `STI_used.csv` | STI daily PI, RI (total-return index, the benchmark), DY, MV. |
| `STI_membership_used.csv` | STI membership periods (SGX code, name, entry date, last date in the index, notes). Notes marked 估計 / 推定 are estimated dates. |
| `tickers.csv` | Datastream code → company name. |
| `mas_fx_daily.csv` | MAS daily noon exchange rates (SGD per unit of USD, CNY, HKD, MYR, PHP, THB). |

The 51 candidates are every company that was an STI constituent at any time since 2010: 40 current or former constituents still listed,
plus 11 that were delisted or removed (COSC, SMTC, NOLS, CMA, OLAM, NOBE, CAPT, JSTG, DBSL, MGLP, SGPS).

## 3. How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_stage1.py         # ~1.5 min
python run_stage2.py         # ~15 min  (writes outputs/frozen_params.json)
python run_stage3.py         # ~5 min   (test period is run once with the frozen parameters)
python run_freq_compare.py   # ~5 min
python -m pytest tests -q    # ~20 min
```

Seeds are fixed (`config.SEED`), so results are reproducible. Stage 2 deletes all data after 2026-09-30 before computing anything,
so nothing from the test period can influence the frozen parameters. `tests/test_freq_compare.py` compares the frozen file with
`outputs/logs/frozen_params_snapshot.json`, a copy taken right after Stage 2.

## 4. Methodology

### 4.1 Universe and data handling
- Point-in-time constituents: on each signal date only the STI constituents in force that day are eligible. Index reviews take effect on a Monday; that Monday's closing signal already uses the new list. Changes are announced in advance, so there is no look-ahead. A holding that leaves the index is sold at the next rebalance; a new constituent can be bought from the first signal date after it joins.
- Tradable filter (each signal date): current constituent, price and market value available, traded that day, and traded on at least 5 of the last 20 trading days. About 30 stocks are tradable each week.
- Delisting: Datastream keeps repeating the last price after a delisting, so each delisted stock is cut at L_i, its last day with trading volume. A position held at L_i is converted to cash at that close (no trading cost); returns inside a holding period end at the L_i close.
- Currency: HKLH, JMTS, DAIR, HPHH and JSTG are quoted in USD and converted with the MAS daily noon rate (closing prices with the same day's rate, opening prices with the previous day's rate). Foreign-currency EPS forecasts and dividends (e.g. THBE, YSHL, CYCL, NOBE) use the latest rate on or before the previous day.
- Returns use RZ (Datastream total-return index). Trades execute at the open: RZ_open = RZ × PO / P.

### 4.2 Calendar and execution
- Rebalance every Tuesday at the open (next trading day if Tuesday is a holiday). The signal date is the previous trading day; factors, scores and volatility forecasts use data up to the signal-date close only.
- A rebalance day is split into prior close → open (old weights) and open → close (new weights). Transaction cost = 15 bps × (Σ|Δw| + |Δcash|).
- A stock with no trade at the rebalance open (suspension) keeps its drifted weight; if it was meant to be sold, the lowest-scored new purchase is skipped so the portfolio still holds K stocks.

### 4.3 Factors, preprocessing and screen
- 19 daily factors: short-term reversal (rev5, mom20, overnight20, intraday20), momentum (mom12_1, mom60), value (dy, ep_ttm, fep), analyst revision (rev), risk (vol20, beta60, mdd60, down60, max20, ivol60), liquidity (amihud20, turnover20) and size.
- Each signal date, each factor: keep tradable constituents → MAD winsorising (±3 robust SD) → z-score → neutralise against sector dummies and standardised ln(market value) → multiply by the expected sign. NaN is filled with 0 only as model input.
- Weekly Spearman Rank IC against the next holding-period return over the training period 2011–2022, with Newey–West t-values. Factors with NW t > 2 enter the model: mom12_1, rev5, vol20, dy, fep, rev, ep_ttm, overnight20 (8 factors), fixed for the whole sample.

### 4.4 Score model: rolling Lasso with cost-aware λ selection
On each signal date a Lasso (coefficients ≥ 0, no intercept) is re-estimated on the previous 260 weeks (minimum 156).
The target y is the next-period return minus the cross-sectional mean; every y in the window ends before the signal-date close.

- λ relative to λ_max: λ_max = max_k (X_k′y / n) is the smallest penalty that sets every coefficient to zero. Candidate penalties are λ = c × λ_max(front part of the window), c ∈ `np.logspace(-3, 0, 30)` without c = 1 (29 values). The final fit on the whole window uses the same c times λ_max(whole window), so the relative penalty is consistent and the re-fit never collapses to all zeros.
- Selection criterion = after-cost strategy performance: on the last 104 weeks of the window, the front-part coefficients score the stocks; a simplified portfolio (top 12, equal weight, rank buffer 15, 15 bps cost on weight changes) is simulated, and the c with the highest after-cost Sharpe ratio of weekly excess returns is chosen.
- Ties are broken at the middle of the tied range.
- Score = X·β̂. Comparison models: Elastic Net, Lassos with a 156-week or expanding window, an equal-weight five-factor score, and fep alone.

### 4.5 Portfolios (8 comparison groups, long-only)

| Code | Stocks | Stock weights | Total exposure |
|---|---|---|---|
| B0 | STI index | — | 100% |
| B0′ | all tradable constituents | market-cap | 100% |
| B1 | all tradable constituents | equal | 100% |
| A1 | all tradable constituents | 1/σ̂ | 100% |
| A2 | all tradable constituents | 1/σ̂ | volatility target |
| S1 | top K by score | equal | 100% |
| S2 | top K by score | 1/σ̂ | 100% |
| S3 | top K by score | 1/σ̂ | volatility target |

- Selection: top K by score; a holding is kept while it ranks within the top K + 3.
- σ̂ᵢ: 60-day EWMA (λ = 0.94), floored at the cross-sectional 5th percentile.
- Caps (all groups except B0′): 15% per stock; sector weight ≤ the sector's market-cap weight among constituents + 10 percentage points (Real Estate uncapped). No-trade band of 2 percentage points.
- Volatility target (A2, S3): GARCH(1,1) with Student-t errors on the portfolio's past 500 daily returns forecasts the 5-day volatility σ̂_p; exposure = min(1, target / σ̂_p), the rest in cash. No leverage.

### 4.6 Volatility experiments and parameter choice (Stage 2)
Starting from a baseline, one setting is changed at a time:

| Experiment | Question | Options (baseline in bold) | Evaluated on |
|---|---|---|---|
| E1 | How to estimate σ̂ᵢ | 20/60-day SD, **EWMA 0.94**, EWMA 0.97, Parkinson 20-day, stock GARCH | S2 |
| E2 | Strength of the low-volatility tilt w ∝ σ̂^(−γ) | γ = 0, 0.5, **1**, 2 | S2 |
| V1 | Volatility target | 10%, **12%**, 15% (plus a no-target control) | S3 |
| K | Number of stocks | 8, 10, **12**, 15 | S3 |
| SECTOR_OVER | Sector over-weight cap | **0.10**, 0.15 | S3 |

Selection rule (validation only, 2023-01 to 2026-07): an option replaces the baseline only if it raises the validation Sharpe ratio by at least 0.05 and also beats the baseline in the training period (2014–2022); otherwise the baseline is kept.

### 4.7 Frozen parameters (`outputs/frozen_params.json`)

| Parameter | Value |
|---|---|
| Factors in the model | mom12_1, rev5, vol20, dy, fep, rev, ep_ttm, overnight20 |
| Model | rolling Lasso, 260-week window, λ = c × λ_max chosen by after-cost performance |
| K / rank buffer | 15 / 3 |
| σ̂ᵢ / γ | EWMA 0.94 / 1 |
| Caps | 15% per stock, sector + 10 percentage points |
| Volatility target | 15%, GARCH(1,1)-t, 500-day lookback |
| Cost | 15 bps |

The selection rule (see 4.6) chose K = 15 and a 15% volatility target; all other settings stayed at the baseline.

### 4.8 Periods

| Period | Dates | Use |
|---|---|---|
| IC tests / factor screen | 2011–2022 | choosing the 8 factors |
| Training (backtest) | 2014-01-07 – 2022-12 | first Lasso score on 2014-01-06 |
| Validation | 2023-01 – 2026-07 | the only period used to choose parameters |
| Freeze | 2026-09-30 | parameters written to `frozen_params.json` |
| Test | 2026-10-01 – 2026-10-29 | built at the 10/1 open, rebalanced 10/6, 10/13, 10/20, 10/27; run once |

## 5. Key results

### 5.1 History with frozen parameters (after costs, 2014-01 to 2026-09)

| Portfolio | Training Sharpe | Validation Sharpe | Full-period Sharpe | Full-period annual return | Max drawdown | Weekly turnover |
|---|---|---|---|---|---|---|
| B0 STI | 0.32 | 1.98 | 0.72 | 9.2% | −33.5% | — |
| B0′ cap-weighted | 0.24 | 1.88 | 0.63 | 7.9% | −33.5% | 0.6% |
| B1 equal | 0.20 | 1.45 | 0.47 | 5.8% | −35.8% | 0.5% |
| A1 1/σ | 0.17 | 1.38 | 0.43 | 5.2% | −34.5% | 1.2% |
| A2 1/σ + vol target | 0.17 | 1.40 | 0.45 | 4.8% | −24.9% | 1.7% |
| S1 selected, equal | 0.23 | 1.52 | 0.50 | 6.6% | −38.3% | 17.0% |
| S2 selected, 1/σ | 0.20 | 1.48 | 0.46 | 5.9% | −38.4% | 17.9% |
| S3 selected, 1/σ + vol target | 0.20 | 1.49 | 0.49 | 5.5% | −29.6% | 17.9% |

- No portfolio beats the STI after costs. Before costs, S3's full-period Sharpe ratio is 0.77 (STI 0.72); trading costs (about 2.9% a year) remove the edge.
- Stock selection adds value over the equal-weight universe (S1 0.50 vs B1 0.47; validation 1.52 vs 1.45), but not over the cap-weighted universe (B0′ 0.63).
- 1/σ weighting does not help (A1 vs B1, S2 vs S1).
- The volatility target mainly cuts drawdowns: −24.9% for A2 vs −34.5% for A1, −29.6% for S3 vs −38.4% for S2.
- Lasso score, weekly Rank IC: 0.033 (t = 3.32) in training, 0.007 (t = 0.48) in validation, so there is little out-of-sample predictive power.

### 5.2 Test period (2026-10-01 to 10-29, cumulative return after costs)

| B0 STI | B0′ | B1 | A1 / A2 | S1 | S2 / S3 |
|---|---|---|---|---|---|
| −4.92% | −4.78% | −4.08% | −4.29% | −4.68% | −4.64% |

The forecast volatility stayed below 15%, so exposure was 100% throughout (S3 = S2, A2 = A1). Twenty trading days are too few to draw conclusions.

### 5.3 Sensitivity (S3, full-period Sharpe after costs; frozen = 0.49)

| Variant | Sharpe |
|---|---|
| Cost 10 / 20 / 30 bps | 0.58 / 0.40 / 0.22 |
| K = 8 / 10 / 12 | 0.18 / 0.28 / 0.29 |
| Volatility target 10% / 12% | 0.39 / 0.44 |
| Tuesday-close execution | 0.51 |
| Excluding THBE and YSHL | 0.36 |
| Excluding IHHH and EMPE | 0.46 |
| Fixed 38-stock universe (survivors only, no membership filter) | 0.45 |

### 5.4 Weekly vs monthly rebalancing (control experiment, parameters unchanged)
Monthly S3 reaches a full-period Sharpe ratio of 0.59 against 0.49 weekly, mainly through lower costs (1.0% vs 2.9% a year),
with a larger maximum drawdown (−35% vs −30%). The main setting stays weekly.

## 6. Tests (`tests/`)
- Look-ahead: perturbing all raw data after a date d must leave factors, scores, weights, exposure and NAV up to d unchanged (several cut-offs, including one before a delisting and a constituent change).
- Calendar and execution: rebalance after the signal date, Tuesday rule, open-price execution, RZ_open = RZ when PO = P.
- Model: every training y ends before the signal date; the λ validation block lies inside the window; λ = c × λ_max always leaves at least one non-zero coefficient; tie-breaking and the after-cost simulation are checked on hand-computed examples.
- Portfolio and backtest: weights sum to exposure, no negative weights, caps respected, hand-calculated NAV, delisting converted to cash, no holdings outside the universe.
- Data truncation: no Stage 2 output contains dates after 2026-09-30.

## 7. Outputs (`outputs/`)

| File | Content |
|---|---|
| `frozen_params.json` | frozen parameters (Section 4.7) |
| `tables/perf_frozen_history.csv` | performance of the 8 groups with frozen parameters (training / validation / full, gross / net) |
| `tables/perf_test.csv`, `nav_test.csv`, `weights_test.csv`, `rebal_test.csv`, `trades_test.csv` | test-period results |
| `tables/test_scores.csv`, `lasso_coef_test.csv` | test-period scores and Lasso coefficients |
| `tables/nav_frozen_history.csv`, `exposure_frozen.csv` | daily NAV and weekly exposure with frozen parameters |
| `tables/sensitivity.csv` | sensitivity tests (Section 5.3) |
| `tables/param_selection.csv`, `vol_experiments.csv`, `vol_target_2020.csv` | volatility experiments and parameter choice |
| `tables/perf_main.csv`, `perf_models.csv`, `model_ic.csv` | baseline-setting results, comparison models, score IC |
| `tables/lasso_log.csv`, `lasso_coef.csv`, `lasso_selection_freq.csv` | Lasso fits (λ_max, c, number of non-zero coefficients), coefficients, selection frequency |
| `tables/ic_summary_train.csv`, `ic_decay_train.csv`, `factor_corr_train.csv`, `factor_coverage.csv` | Stage 1 factor tests |
| `tables/membership_count.csv`, `membership_events.csv`, `delisted_Li.csv`, `exclusion_sales.csv` | universe checks |
| `tables/diagnostics.csv`, `brinson_summary.csv` | weight diagnostics, Brinson sector attribution (S3 vs B0′) |
| `tables/freq_compare.csv`, `freq_compare_lasso.csv` | weekly vs monthly control experiment |
| `tables/assets_daily_close.csv` | submission file: daily closing prices (SGD) of the test-period constituents |
| `figures/` | NAV, drawdown, exposure, gross vs net, γ curve, Lasso heatmap, S3 sector weights, IC decay, weekly vs monthly |
| `logs/frozen_params_snapshot.json` | copy of the frozen parameters taken right after Stage 2 |

`outputs/data/` (intermediate parquet files) is not included; it is rebuilt by the scripts.

## 8. Limitations and disclosures
1. Factor screen period: the factor screen uses 2011–2022 IC, which overlaps the training backtest, so training-period Sharpe ratios may be overstated by up to about 0.1 (compared with a Lasso on all 19 factors). Validation and test results are unaffected.
2. Short validation and test periods: validation is 3.6 years (Sharpe ratio estimation error about ±0.5); the test period has 20 trading days. The large validation improvement from K = 12 to K = 15 may partly be noise.
3. Costs and turnover: the S groups turn over about 18% a week; costs explain most of the gap to the STI.
4. Delisting proceeds are approximated by the last closing price (e.g. CapitaLand's restructuring paid CLI shares plus cash and CLCT units; Jardine Strategic was bought out at USD 33 per share).
5. Membership dates: five change dates are estimates and are used as given.
6. Data: USD closing prices and the MAS noon rate are about five hours apart; NOBE's dividend currency is inferred; the market-cap weights use full market value, not the STI's free-float weights.
7. r_f = 0: cash earns nothing, which understates the return of the volatility-target portfolios when they hold cash.
