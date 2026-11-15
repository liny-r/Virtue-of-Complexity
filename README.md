# The Virtue of Complexity Everywhere — a reduced replication

**Paper:** Bryan Kelly, Semyon Malamud, Kangying Zhou (2022), *The Virtue of Complexity Everywhere*,
SSRN Working Paper 4166368 (version of July 18, 2022). Theory comes from the companion paper,
Kelly, Malamud, Zhou (2022), *The Virtue of Complexity in Return Prediction*. The paper has no
public code that I used; everything here is written from the paper's equations.

Course: Deep Learning in Finance, Assignment 1.

## In two minutes

The paper fits a ridge regression on **P random Fourier features** of a few return-predicting
signals, with P far larger than the training window T (as small as 12), and reports that
out-of-sample **Sharpe rises with complexity c = P/T** once the ridge penalty z is large enough.

I implemented that model and tested it three ways: on a simulator where the theory's assumptions
hold, on six real equity factors, and with two controls the paper does not report.

| Question | Answer in this reduced replication |
|---|---|
| Do the theory's predictions hold in a simulator built to satisfy them? | Mostly. R² explodes at c = 1 and recovers; with ridge, Sharpe rises monotonically in c. The Sharpe *dip* at c = 1 appears only when the ridge is ≈ 0 (z ≤ 1e-8), not at the paper's smallest z = 1e-3. |
| Does Sharpe rise with c on real factors? | At T = 12 with enough ridge, yes: 0.32 → 0.41 (annualised, mean of 6 factors), about 1.3× and not the ~3× the paper reports. At T = 60 and 120 the large-ridge curve is **flat**, unlike the paper's Figure 5. |
| Does the model beat simple alternatives? | It beats a linear model (0.41 vs ≈ 0.2) but **not** staying long the (vol-scaled) factor (0.46), and it is close to a trailing 12-month mean (0.40). |
| Is the rise with c evidence of learned non-linear predictability? | Not by itself. With the signals **shuffled in time** the curve still rises (0.19 → 0.37). The rise is the random features converging to the exact kernel (verified: exact kernel ridge matches c = 500 to 3 decimals). Real signals add about +0.04 at high c (≈ 2.3 placebo sd). |

All results are for 6 factors, 1931–2021 (1968–2021 for RMW, CMA), not the paper's 159 factors,
44 futures and Treasuries. See "What I did not do".

## What I implemented

* `voc/features.py` — random Fourier features for the mixture of 7 Gaussian kernels, eqs. (36)–(37).
* `voc/ridge.py` — rolling ridge, eqs. (8)–(11), in the dual (T×T) form so P = 7,000 is cheap, with all
  nested models L = 2…P obtained in one pass (cumulative Gram matrices over blocks of features).
* `voc/metrics.py` — out-of-sample R², expected return, volatility, Sharpe, ‖β‖, eqs. (41)–(46).
* `voc/preprocess.py` — return and signal standardisation, eqs. (13)–(16), strictly past-only.
* `voc/simulate.py` — a simulator with a known non-linear conditional mean (paper eq. 1).
* `voc/kernel.py` — exact kernel ridge (the c → ∞ limit), used as a mechanism check.
* `experiments/` — simulator check, real-factor curves, placebo test, exact-kernel check.
* `tests/` — 7 tests (see below).

## What I did not do

* **Futures and Treasuries.** The paper's futures come from the paid Stevens Continuous Futures
  feed and its bonds from CRSP/Fama–Bliss. I did not substitute other data.
* **159 factors.** I use 6 (Mkt-RF, SMB, HML, RMW, CMA, Mom) from Kenneth French's library, sample cut
  at 2021-12 to match the paper. The 153 JKP factors and the 15 Welch–Goyal predictors are not used.
* **Only the 2-signal "momentum" specification** (lagged return, trailing 11-month mean), not the
  15-signal Welch–Goyal one.
* **20 random-feature seeds**, not 100, and per-seed metrics are averaged as in eqs. (47)–(51).
  The seed-averaged position of eq. (39) is not implemented.
* **P up to 7,000 columns**, not 12,000. (c ≤ 500 at T = 12.)
* No transaction costs, as in the paper.

## Install and run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py --quick     # reduced run, exercises the whole pipeline (~2 min)
python run_all.py             # full run, regenerates everything reported below (~45 min on a laptop;
                              # the T = 120 robustness run alone took ~35 min)
```

`run_all.py` runs the tests, downloads the factor data if `data/ff_factors_monthly.csv` is missing
(`data/fetch_data.py`; the small CSV is committed), then each experiment. Random seeds are fixed.
Quick mode writes `*_fast` files and is not the reported result.

## Results

### 1. Simulator (50 simulated datasets, T = 12, 5 random-feature draws each)

Returns have unit variance and a small non-linear predictable part (snr = 0.05).

| Prediction | Result |
|---|---|
| (1) R² negative while Sharpe positive | Yes. R² ≈ −1.9 at c = 1 for z = 1e-3; Sharpe positive at all c. |
| (2) Sharpe dips toward 0 at c = 1 without ridge | **Only for z ≤ 1e-8** (Sharpe 0.006 at c = 2/3 → 0.001–0.002 at c = 1 → 0.014 at c = 2). Not at z = 1e-3 (paired difference c = 1 vs 2/3: +0.003, t = +2.1, i.e. no dip). |
| (3) Sharpe monotone in c with enough ridge | Yes, every z in 1e-3…1e3 (0.006 → 0.034 at z = 1e-3; 0.005 → 0.014 at z = 1e3). |
| Lemma 1: ‖β‖ falls in L past c = 1 | Yes for small z (≈ 9 → 6), plateauing instead of going to 0. |

Adding an unconditional drift (Sharpe 0.08 per period) did not create a dip at z = 1e-3
(`figures/sim_theory_mu0.08.png`). Figures: `figures/sim_theory.png`, `..._z-12_-8_-4_-3.png`.

### 2. Real factors (headline) — `figures/real_factors_T12.png`

Mean annualised OOS Sharpe over the 6 factors, T = 12, 20 seeds, up to 7,000 features:

| | c = 0.17 | c = 1 | c = 16 | c = 500 |
|---|---|---|---|---|
| z = 1e-3 | 0.244 | 0.175 | 0.160 | 0.197 |
| z = 1e-1 | 0.300 | 0.359 | 0.388 | 0.393 |
| z ≥ 10 | 0.316 | 0.392 | 0.411 | 0.412 |

| Baseline (not reported in the paper) | Sharpe |
|---|---|
| Always long the vol-scaled factor | **0.461** |
| Trailing 12-month mean of returns | 0.396 |
| Linear ridge on the 2 signals | 0.10 – 0.21 (depending on z) |

Mean OOS R² at c = 1 is −0.93 for z = 1e-3 and −0.02 for z = 1e3: R² and Sharpe disagree, as the
theory says. Curves for z ≥ 10 coincide exactly: Sharpe is invariant to the scale of the position,
so at large z only the direction of β matters. The paper's choice z = 1000 is therefore the
infinite-ridge limit, in which the position is a kernel-weighted average of the last T returns.

### 3. Placebo: shuffle the signals in time — `figures/placebo.png`

20 permutations of the signal rows (targets keep their true order), 5 feature draws each.

| z = 1e3, c | Real | Placebo (sd) | Real − placebo |
|---|---|---|---|
| 0.17 | 0.316 | 0.185 (0.018) | +0.13 |
| 1 | 0.392 | 0.369 (0.013) | +0.02 |
| 16 | 0.411 | 0.373 (0.016) | +0.04 |
| 500 | 0.412 | 0.375 (0.016) | +0.04 |

Gain from c = 0.17 to 500: real +0.096, placebo **+0.189**. At z = 1e-3 real and placebo overlap for
c ≥ 1 (0.5–1.2 sd). The sd is across shuffles of one dataset, so "sd units" are descriptive, not a p-value.

### 4. Exact kernel — mechanism check

Exact mixture-kernel ridge (no random features) gives, at z = 1e3, 0.412 on real signals and 0.375
on shuffled ones, matching the random-feature model at c = 500 (0.412, 0.375). So the "complexity"
axis here is how well the sampled kernel approximates the exact one.

### 5. Training window T = 60 and 120 (paper's Figure 5 analogue)

Same 6 factors, 5 feature draws, up to 7,000 features. Mean annualised OOS Sharpe
(`figures/real_factors_T60.png`, `..._T120.png`; logs in `figures/*.log`):

| T | z | c = 0.17 | c = 1 | c = 16 | c_max |
|---|---|---|---|---|---|
| 60 | 1e-3 | 0.294 | 0.246 | 0.257 | 0.262 (c = 64) |
| 60 | 1e-1 | 0.411 | 0.435 | 0.439 | 0.439 |
| 60 | 1e3 | 0.407 | 0.416 | 0.415 | 0.416 |
| 120 | 1e-3 | 0.381 | 0.335 | 0.348 | 0.351 (c = 32) |
| 120 | 1e-1 | 0.506 | 0.514 | 0.517 | 0.518 |
| 120 | 1e3 | 0.465 | 0.461 | 0.461 | 0.462 |

| Baseline | T = 60 | T = 120 |
|---|---|---|
| Always long | 0.466 | 0.490 |
| Trailing T-month mean | 0.390 | 0.418 |
| Linear ridge on the 2 signals | 0.24 – 0.29 | 0.33 – 0.37 |

With a large ridge the curve is **flat** at both windows (+0.009 from c = 0.17 to 64 at T = 60;
−0.003 from c = 0.17 to 32 at T = 120). The paper's Figure 5 reports a strictly monotone rise at
both windows. Caveats: c = 0.17 means more features at longer T (10 at T = 60, 20 at T = 120,
versus 2 at T = 12), so the low-complexity end is not comparable across windows; the OOS samples
also start later (1935, 1940). I did not run the placebo at T = 60/120.

At T = 120 the intermediate ridge z = 1e-1 gives 0.518, above always-long (0.490). That z was
picked by looking at the out-of-sample grid, so I do not count it as evidence of timing skill.

### Side by side with the paper

| | Paper (Fig. 3, Panel B, 159 factors) | Here (6 factors) |
|---|---|---|
| Sharpe, low complexity, large z | ≈ 0.1 | 0.32 (c = 0.17) |
| Sharpe, high complexity, large z | ≈ 0.30 | 0.41 |
| Gain from low to high c | "a factor of three" (text, p. 26) | ≈ 1.3× |
| No-ridge curve (z = 1e-3) | dip near c = 1 (read off the figure) | falls from 0.244 (c = 0.17) to a minimum of 0.147 at c = 4, then recovers to 0.197; the minimum is not at c = 1 |

The paper's values are read off its figure by eye, and the paper does not say whether Sharpe
ratios are annualised (I annualise). The two factor sets differ (the six here are the best-known
factors), so the levels are not directly comparable; the *shape* and the *baselines* are the point.

## What didn't work, or didn't match

1. **The size of the effect.** About 1.3× gain in Sharpe, not ~3×. I do not know whether this is the
   smaller factor set, the 2-signal specification, my scaling choices, or the sample.
2. **No demonstrated timing edge.** The best model (0.412) is below always-long (0.461). The gap is
   within noise (a single factor's annualised Sharpe has a standard error of about 0.1), so this
   does not show the model *loses*; it shows the paper's evidence does not include a baseline that
   would let one say it wins.
3. **The double-ascent dip.** In the simulator it needs z ≤ 1e-8. At the paper's z = 1e-3 my
   feature scaling (1/√L, see below) shows no dip at c = 1 on simulated data. So either the
   paper's scaling differs from mine or its dip has another cause. **Unresolved.**
4. **The rise with c is mostly there without any signal information** (placebo, section 3), and it
   **does not reproduce at T = 60 or 120** for large z (section 5), where the paper claims it does.
5. **Untested explanations.** (a) Why the placebo converges to ≈ 0.375: I suspect the wide
   bandwidths (γ = 8, 16) make the kernel nearly constant on standardised signals, so the position
   is close to a trailing mean; the placebo level is not equal to the trailing mean (0.396), so this
   is at most part of the story. (b) Why very small models do poorly: I suspect random-sign kernel
   weights at L = 2. Neither was tested.
6. **Not done:** see "What I did not do". All conclusions are limited to 6 factors and one
   signal specification.

## Where the paper is ambiguous, and what I chose

| Item | Choice |
|---|---|
| **Feature scaling** | Paper: S = P^{-1/2} h. I scale by the width L of the model being fitted (G = G_raw / L). With a fixed 1/√P_total, "z = 1e-3" is not near-ridgeless at small L and R² does not explode at c = 1. This is an **inference**, not stated in the paper. |
| **Eq. (36) vs (37)** | Taken literally, eq. (37) (frequencies γ·ω, ω ~ N(0, I)) gives a kernel exp(−γ²‖Δx‖²/2), not eq. (36)'s exp(−‖Δx‖²/(2γ)). I follow eq. (36): frequencies ω/√γ. A `convention="paper"` option reproduces the literal version. |
| **Number of features** | Each of the P = 7p weights gives a (sin, cos) pair, so 2P columns; L counts columns. |
| **Eq. (15)** | The sum over τ = 1…t−37 read literally drops the first 36 and most recent observations. I use an expanding RMS of data strictly before t with ≥ 36 observations. |
| **Annualisation** | Not stated for Fig. 3. I annualise monthly Sharpe by √12. |
| **Timing** | Signals at t, target y[t] = r[t+1]/σ_t, trained on pairs t−T…t−1 (all known at t). `tests/` checks that changing future data does not change a past forecast. |

## Tests

`pytest -q` runs 7 tests: the dual-form ridge equals the paper's eq. (9) computed literally (both
scalings); no look-ahead; random features approximate the mixture kernel; the ridgeless solution
interpolates and has minimum norm (Lemma 1); metrics; exact kernel ridge equals a many-feature
random-feature model.

Data check: the six downloaded factors reproduce the paper's Table 1 closely (e.g. Mom: 7.51% mean,
16.35% vol, SR 0.46 vs the paper's 7.70%, 16.46%, 0.47; Mkt-RF 8.33%, 18.47%, 0.45 vs 8.29%, 18.63%, 0.45).

## Layout

```
voc/            reusable modules (features, ridge, kernel, metrics, preprocess, simulate)
experiments/    sim_theory, real_factors, placebo, exact_kernel
data/           fetch_data.py + ff_factors_monthly.csv (small, committed)
figures/        generated figures and .npz result arrays
tests/          unit tests
run_all.py      single entry point
AI_USAGE.md     AI usage disclosure
```
