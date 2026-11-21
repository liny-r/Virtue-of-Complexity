# The Virtue of Complexity Everywhere — a reduced replication

**Paper:** Bryan Kelly, Semyon Malamud, Kangying Zhou (2022), *The Virtue of Complexity Everywhere*,
SSRN Working Paper 4166368 (version of July 18, 2022). Theory comes from the companion paper,
Kelly, Malamud, Zhou (2022), *The Virtue of Complexity in Return Prediction*. The paper has no
public code that we used; everything here is written from the paper's equations.

Course: Deep Learning in Finance, Assignment 1.

## In two minutes

The paper fits a ridge regression on **L random Fourier features** of a few return-predicting
signals, with L far larger than the training window T (as small as 12), and reports that
out-of-sample **Sharpe rises with complexity c = L/T** once the ridge penalty z is large enough.

We implemented that model and replicated the paper's equity-factor results (Figures 1–5) on its own
universe: **159 US equity factors** (6 French-library factors + 153 JKP characteristic factors), monthly,
for **both** of the paper's equity studies: the 15 Welch–Goyal macro predictors (its Panel A) and the 2
factor-momentum signals (its Panel B), T = 12, 60, 120, 100 feature draws at T = 12. Beyond the paper, we tested
the theory in a simulator and ran controls the paper does not report (baselines, a placebo, exact kernel).

| Question | Answer |
|---|---|
| Does Sharpe rise with c on the paper's 159 factors (Fig. 3)? | **Yes, at T = 12.** Annualised Sharpe, mean of 159 factors, large ridge: Welch–Goyal signals 0.117 → 0.368 (3.1×), momentum signals 0.166 → 0.342 (2.1×); the paper's curves go ≈ 0.11 → 0.36 and ≈ 0.1 → 0.30. 96–98% of factors improve. |
| Do the other panels match? | Welch–Goyal: the Sharpe curve at low ridge matches closely (dip to 0.08 at c = 1, the paper ≈ 0.07; 0.305 at c = 500, the paper ≈ 0.32), and so do the expected return (0.038 vs ≈ 0.041) and volatility (0.42 vs ≈ 0.45). **Not matched:** ‖β‖ falls with c in the paper and plateaus here, and the curves for z = 1…100 sit lower in the paper than here (section 4). |
| And at T = 60 and 120 (Fig. 5)? | A smaller rise: momentum +0.068 (T = 60) and +0.018 (T = 120), Welch–Goyal +0.069 and +0.030, at large ridge (10 draws). |
| Beyond the paper: does the model beat simple alternatives? | **No.** It beats a linear model but not staying long the (vol-scaled) factor: T = 12, momentum 0.342 vs 0.353 (−0.011, s.e. 0.017); Welch–Goyal 0.368 vs 0.349 (+0.019, s.e. 0.018). |
| Beyond the paper: is the rise with c learned predictability? | **Mostly not.** With the signals **shuffled in time** the curve rises by the same amount (+0.177 for the placebo vs +0.176 for real signals, T = 12); the rise is the random features converging to the exact kernel. Real signals add a roughly constant +0.02 at high c (≈ 4 placebo sd) at T = 12, and more at longer windows (+0.05 at T = 60, +0.07 at T = 120, 9–13 sd). |

Results are for US equity factors only: no futures, no Treasuries. See "What we did not do".

## References

* Kelly, B., Malamud, S., Zhou, K. (2022). *The Virtue of Complexity Everywhere*. SSRN 4166368.
* Kelly, B., Malamud, S., Zhou, K. (2022). *The Virtue of Complexity in Return Prediction*. Journal of Finance (companion theory).
* Jensen, T. I., Kelly, B., Pedersen, L. H. (2023). Is There a Replication Crisis in Finance? *Journal of Finance* 78(5); SSRN 3774514. Defines the 153 factors; data from jkpfactors.com (`bkelly-lab/jkp-data`).
* Welch, I., Goyal, A. (2008). A Comprehensive Look at the Empirical Performance of Equity Premium Prediction. *Review of Financial Studies* 21(4). Predictor data: Amit Goyal's website.
* Fama–French and momentum factors: Kenneth French's Data Library (Dartmouth Tuck).
* Rahimi, A., Recht, B. (2007). Random features for large-scale kernel machines (the feature construction).

## What we implemented

* `voc/features.py` — random Fourier features for the mixture of 7 Gaussian kernels, eqs. (36)–(37).
  The default (`convention="paper"`) is the literal eq. (37): frequencies γ·ω. `convention="kernel"` is eq. (36).
* `voc/ridge.py` — rolling ridge, eqs. (8)–(11), in the dual (T×T) form so L = 7,000 is cheap, with all
  nested models L = 2, 4, … obtained in one pass (cumulative Gram matrices over blocks of features).
* `voc/metrics.py` — out-of-sample R², expected return, volatility, Sharpe, ‖β‖, eqs. (41)–(46).
* `voc/preprocess.py` — return and signal standardisation, eqs. (13)–(16), strictly past-only; the 2 momentum
  signals and the 15 Welch–Goyal predictors.
* `experiments/curves.py` — the paper's Figures 1–5 analogues on the 159 factors (the replication);
  `--signals momentum|wg`, `--T`, `--seeds`, `--convention`.
* `experiments/check_table1.py` — our data against the paper's Table 1.
* `experiments/report_numbers.py` — prints every number quoted below from the saved `.npz` results.
* `experiments/kernel_check.py` — random features vs the exact kernel; what the kernel looks like; why the placebo matches.
* Beyond the paper: `voc/simulate.py` + `experiments/sim_theory.py` (simulator with a known non-linear
  mean), `experiments/placebo.py`, `voc/kernel.py` + `experiments/exact_kernel.py` (exact kernel ridge,
  the c → ∞ limit), `experiments/scaling_check.py` (feature normalisation).
* `tests/` — 13 tests (see below).

## What we did not do

* **Futures and Treasuries.** The paper's futures come from the paid Stevens Continuous Futures
  feed and its bonds from CRSP/Fama–Bliss. We did not substitute other data.
* **Seeds at T = 60 and 120:** 100 feature draws at T = 12 (as in the paper), but 10 at T = 60/120 and in the scaling
  check. A 100-draw check at T = 60 is in section 2. Per-seed metrics are averaged as in eqs. (47)–(51); the seed-averaged position of eq. (39) is not
  implemented. The smallest models (L = 2, 4) depend strongly on the draw (section 3.4).
* **L up to 7,000 features (columns)**, not 12,000: c ≤ 500 at T = 12 (the paper reaches 1,000), c ≤ 64 at T = 60 and
  c ≤ 32 at T = 120.
* The README figures use a log c axis. The paper uses a linear axis to c = 50 with a break to 1,000: the notebook (section 3c) and the slides redraw the results on those axes (our largest model is c = 500).
* One of the 159 factors (`intrinsic value`) is a substitute (see section 0).
* The Welch–Goyal predictors are our reconstruction from Goyal's public spreadsheet (a 2026 vintage) and the paper's
  footnote list; the paper's exact variable construction is not given (section "Where the paper is ambiguous").
* No transaction costs, as in the paper.

## Install and run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m experiments.curves --workers 8                   # momentum signals, T = 12 (~3 min with 20 seeds; ~20 min with --seeds 100)
python -m experiments.curves --signals wg --workers 8      # Welch–Goyal signals
python run_all.py --quick     # reduced run, exercises the whole pipeline (< 3 min)
python run_all.py             # full run, regenerates everything reported below (~4 h on 8 cores)
python run_all.py --eq36      # additionally the eq. (36) sensitivity set (~2.5 h more; files end in _eq36)
python -m experiments.plot_paper_axes   # redraw saved results on the paper's axes (slides; no re-run)
```

Tested on Python 3.13 (macOS, 8 cores). `--workers 1` (the default for the single scripts) gives the same
numbers more slowly: seeds are fixed and the per-factor jobs are independent. The
notebook `notebooks/virtue_of_complexity.ipynb` can be opened and run top to bottom after
`pip install -r requirements.txt`; it re-plots the saved results and runs its live checks in seconds.

`run_all.py` runs the tests, downloads the factor data if the CSVs in `data/` are missing
(`data/fetch_data.py`, with `--jkp` and `--wg` for the JKP and Welch–Goyal files; the small CSVs are committed), then each
experiment, and finally `experiments/report_numbers.py`, whose output (`figures/report_numbers.log`) lists every number
below. Random seeds are fixed. Quick mode writes `*_fast` files and is not the reported result.

File tags: no suffix = momentum signals with the literal eq. (37) (main); `_wg` = Welch–Goyal signals; `_eq36` = eq. (36) kernel (sensitivity).

## Results

### 0. Data consistency with the paper (`python -m experiments.check_table1`, `figures/table1_check.png`)

We transcribed the paper's Table 1 (159 factors; `data/paper_table1.csv`) and recomputed the same statistics from
our files, cut at 2021-12. Of 158 factors matched by name, annualised mean, volatility and Sharpe
correlate 0.996, 0.9995 and 0.997 with the paper's. Start months agree for 155 of 158 (`age`, `gp_at`,
`opex_at` start one month earlier here). Sharpe differences average +0.010 (sd 0.022, max 0.117 for
`o_score`), mean-Sharpe over factors is 0.320 here against 0.310 in the paper. The small differences are
what we would expect from a newer data vintage (the JKP file we downloaded is dated 2026-04); we did not test
that. The paper's JKP factors are therefore the **US, capped value-weighted** series, which is what we use.

Checked against Jensen, Kelly & Pedersen (2021/2023), SSRN 3774514, which defines these factors: tercile
long-short spreads with breakpoints from non-micro stocks, capped value weights (market equity
winsorised at the NYSE 80th percentile), one-month holding period, signed as in the original paper,
at least 5 stocks per leg and at least 60 months. The 10% volatility scaling in that paper belongs to its Bayesian analysis;
the downloadable series are unscaled, and the paper's Table 1 volatilities (4%–19%) match the unscaled data.

Two things to know. (1) One of the 159 rows, `intrinsic value`, has no counterpart: the public JKP file has
`ival_me` (intrinsic value to market) instead, and the two differ (paper: mean −1.43%, Sharpe −0.15;
`ival_me`: +3.99%, 0.40). So 152 of our 153 JKP series are the paper's, and one is a substitute; it is
about 0.6% of the average. (2) The factors are signed using the original publications' findings, so
"always long" is a benchmark that uses knowledge of which direction worked in the literature; a real-time
investor would not have it. This makes the always-long baseline (0.349–0.353) flattering to compare against.

### 1. The paper's Figures 1–4 on the 159 factors (T = 12), momentum signals — `python -m experiments.curves --seeds 100`

Mean over 159 factors, 100 feature draws, up to 7,000 features, annualised Sharpe, literal eq. (37) features.
Figures: `figures/fig1_T12.png` (R², ‖β‖), `fig2_T12.png` (expected return, volatility), `fig3_T12.png` (Sharpe),
`fig4_T12.png` (per-factor normalised Sharpe). Mean Sharpe:

| | c = 0.17 | c = 1 | c = 16 | c = 500 |
|---|---|---|---|---|
| z = 1e-3 | 0.121 | 0.093 | 0.176 | 0.213 |
| z = 1e-1 | 0.156 | 0.243 | 0.315 | 0.322 |
| z ≥ 10 | 0.166 | 0.289 | 0.340 | 0.342 |

At c = 500 minus c = 0.17 (z = 1e3): +0.176 (s.e. 0.009 across factors, 96% of factors positive), 2.1×. At
z = 1e-3 the curve falls to a minimum of 0.093 **at c = 1** and recovers to 0.213 at c = 500. Curves for z ≥ 10 coincide: Sharpe is
invariant to the scale of the position, so at large z only the direction of β matters (under our feature normalisation;
see section 4). With 20 draws instead of 100 the c = 0.17 point was 0.135 (not 0.166) and c = 1 0.282 (not 0.289), while c ≥ 16
agreed to 0.001; the smallest models are the noisy part (section 3.4).

**Fig. 1/2 quantities at c = 500** (mean of factors; paper values for its Panel B read off the figure at c ≈ 1000 by eye):

| z | OOS R² | ‖β‖ | ER / month | Vol / month (std) | paper ER | paper Vol |
|---|---|---|---|---|---|---|
| 1e-3 | −0.18 | 7.8 | 0.0393 | 0.63 | ≈ 0.039 | ≈ 0.8–0.9 |
| 1e-2 | −0.11 | 4.9 | 0.0377 | 0.51 | ≈ 0.039 | |
| 1e-1 | −0.022 | 1.4 | 0.0273 | 0.28 | ≈ 0.039 | |
| 1 | −0.010 | 0.23 | 0.0074 | 0.07 | ≈ 0.039 | |
| 1e1 | −0.017 | 0.03 | 0.0009 | 0.009 | ≈ 0.0375 | |
| 1e2 | −0.019 | 0.00 | 0.0001 | 0.001 | ≈ 0.030 | |
| 1e3 | −0.019 | 0.00 | 0.0000 | 0.000 | ≈ 0.010 | |

* **R² (Fig. 1):** reproduces the collapse near c = 1 for small z (−1.67 at c = 1, z = 1e-3) and the recovery
  (−0.18 at c = 500); the paper's z = 1e-3 curve ends near −0.45.
* **‖β‖ (Fig. 1):** **partly.** It peaks near c = 2 (9.6) and then falls to 7.8, as the paper's does (Lemma 1), but it
  plateaus where the paper's falls towards ≈ 1 by c = 50 (section 4).
* **ER and Vol (Fig. 2):** low-z ER matches (0.039 vs ≈ 0.04), Vol is lower (0.63 vs ≈ 0.8–0.9). But the
  paper's ER is still 0.010 at z = 1e3 and 0.030 at z = 1e2, where ours is ≈ 0: the paper's z axis sits roughly
  three decades above ours.
* **Fig. 4 (per-factor normalised Sharpe, z = 1e3):** the cross-factor mean is 0.67 at c = 1 and 0.95 at c = 500; 92% of
  factors end above 0.9 of their own range, 4% below 0.5, and 96% end above their c = 0.17 value. The paper reports the rise is
  uniform across assets; here it is for most factors but not all.

### 1b. The paper's first study: 15 Welch–Goyal predictors (its Panel A) — `python -m experiments.curves --signals wg --seeds 100`

Same 159 factors, same model, signals = the 14 predictors of the paper's footnote plus the market return (section
"Where the paper is ambiguous"), 100 draws. Files end in `_wg`. Mean Sharpe:

| | c = 0.17 | c = 1 | c = 16 | c = 500 | paper (by eye) |
|---|---|---|---|---|---|
| z = 1e-3 | 0.079 | 0.079 | 0.231 | 0.305 | dip ≈ 0.07 at c = 1; ≈ 0.29 at c = 50; ≈ 0.32 at c = 1000 |
| z = 1e-1 | 0.106 | 0.215 | 0.329 | 0.353 | ≈ 0.31 at c = 50 (z = 1) |
| z ≥ 10 | 0.117 | 0.265 | 0.358 | 0.368 | ≈ 0.11 → ≈ 0.36 (z = 1e2, 1e3); ≈ 0.345 (z = 10) |

Gain from c = 0.17 to 500 at z = 1e3: +0.250 (s.e. 0.011; 98% of factors positive), 3.1×, which is the paper's "factor
of three". At c = 500: ER 0.038 and Vol 0.42 at z = 1e-3 (paper ≈ 0.041 and ≈ 0.45), OOS R² −0.065 at z = 1e-3 (the paper ≈ −0.05 at c = 50),
‖β‖ 6.1 (paper: falls to ≈ 1 by c = 50). So the Sharpe, ER, Vol and R² panels agree well; ‖β‖ and the z = 1…10 curves
(0.366 here, ≈ 0.31 in the paper at z = 1) do not, as for momentum.

### 2. Training window T = 60 and 120 (paper's Figure 5) — `python -m experiments.curves --T 60 --seeds 10` (and `--signals wg`)

Same 159 factors, 10 feature draws, up to 7,000 features (so c ≤ 64 at T = 60 and c ≤ 32 at T = 120).
Mean annualised Sharpe (`figures/fig3_T60.png`, `fig3_T120.png`, `_wg` versions):

| Signals | T | z | c = 0.17 | c = 1 | c = 16 | c_max |
|---|---|---|---|---|---|---|
| momentum | 60 | 1e-3 | 0.204 | 0.143 | 0.189 | 0.206 (c = 64) |
| momentum | 60 | 1e3 | 0.277 | 0.341 | 0.345 | 0.345 |
| momentum | 120 | 1e-3 | 0.270 | 0.193 | 0.230 | 0.237 (c = 32) |
| momentum | 120 | 1e3 | 0.345 | 0.361 | 0.362 | 0.363 |
| Welch–Goyal | 60 | 1e3 | 0.266 | 0.327 | 0.335 | 0.335 |
| Welch–Goyal | 120 | 1e3 | 0.296 | 0.326 | 0.326 | 0.326 |

| Baseline (not in the paper) | T = 12 | T = 60 | T = 120 |
|---|---|---|---|
| Always long the vol-scaled factor | 0.353 (WG panel: 0.349) | 0.354 (0.349) | 0.351 (0.350) |
| Trailing T-month mean | 0.338 (0.355) | 0.309 (0.323) | 0.303 (0.307) |
| Linear ridge on the signals | 0.11 – 0.16 (0.11 – 0.32) | 0.24 – 0.27 (0.19 – 0.31) | 0.31 – 0.35 (0.22 – 0.31) |

At large ridge the momentum curve rises by +0.068 at T = 60 (s.e. 0.005; 86% of factors positive, 1.24×) and +0.018 at T = 120
(s.e. 0.002; 75%, 1.05×); the Welch–Goyal curve by +0.069 and +0.030. The paper's Figure 5 reports a monotone rise at both windows; we
get that direction but a much smaller size, with the model never clearly above always long: model minus always long is −0.009 (s.e. 0.013) at
T = 60 and +0.013 (0.012) at T = 120 for momentum, −0.014 (0.013) and −0.023 (0.013) for Welch–Goyal.
Caveat: at a fixed c = 0.17 the longer windows already have more features (10 at T = 60, 20 at T = 120, against 2
at T = 12), so the low-complexity end is not comparable across windows. At T = 120 an intermediate ridge z = 1e-1 gives 0.402 (momentum),
above always long (0.351), but that z was picked by looking at the out-of-sample grid, so we do not count it as evidence
of timing skill.

**Robustness: 100 draws at T = 60.** The paper averages 100 feature draws at every window; the main T = 60 runs use 10. We also
ran T = 60 with 100 draws (seeds 1000 + s, so the 10 draws are the first 10 of the 100; results in `figures/curves_T60_100draws.npz`
and `curves_T60_wg_100draws.npz`, logs alongside; `python -m experiments.curves --T 60 --seeds 100`, about 4.5 h per signal set,
which overwrites `curves_T60*.npz`, so copy the existing ones first). Mean annualised Sharpe at large ridge:

| Signals | draws | c = 0.17 | c_max (64) | rise | factors improving |
|---|---|---|---|---|---|
| momentum | 10 | 0.277 | 0.345 | +0.068 (1.24×) | 86% |
| momentum | 100 | 0.295 | 0.345 | +0.050 (1.17×) | 93% |
| Welch–Goyal | 10 | 0.266 | 0.335 | +0.069 (1.26×) | 83% |
| Welch–Goyal | 100 | 0.235 | 0.335 | +0.100 (1.43×) | 86% |

The high end (c ≥ 16) does not move; the smallest models do (by +0.018 and −0.031 at c = 0.17), and the rise with them. Always long, the trailing
mean and the model-minus-always-long gap (−0.009 momentum, −0.014 Welch–Goyal) are unchanged, so the conclusions do not depend on the number of draws; the
size of the rise at T = 60 does. The T = 120 rise (10 draws) should be read the same way.

### 3. Beyond the paper: what the curve does and does not show

**3.1 Baselines the paper omits** (T = 12, momentum signals, mean of 159): always long 0.353, trailing 12-month mean 0.338,
linear ridge 0.11–0.16. The model's best, 0.342, is −0.011 below always long (s.e. 0.017), i.e. no timing edge, and
+0.004 above the trailing mean (s.e. 0.003). The 6 French factors alone give 0.410 at c = 500 (always long 0.461);
the 153 JKP factors give 0.340 (always long 0.349). With Welch–Goyal signals the model is +0.019 above always long
(s.e. 0.018) and +0.013 above the trailing mean (s.e. 0.003): the only comparison here that favours the model, and
within noise of always long. The factors are signed using the original publications' findings (JKP), so "always long"
is a benchmark that uses knowledge of which direction worked in the literature; a real-time investor would not have it.

**3.2 Placebo: shuffle the signals in time** (`python -m experiments.placebo`, `figures/placebo.png`). 20 permutations
of the signal rows (targets keep their true order), one feature draw per shuffle, cycling through the same 100 seeds
as the real run (5 draws per shuffle); z = 1e3, mean over 159 factors:

| c | 0.17 | 1 | 16 | 500 |
|---|---|---|---|---|
| real signals | 0.166 | 0.289 | 0.340 | 0.342 |
| shuffled (mean, sd) | 0.144 (0.057) | 0.263 (0.029) | 0.318 (0.006) | 0.321 (0.006) |

Real minus placebo is +0.02 (≈ 4 sd) at c ≥ 16; at c ≤ 1 it is within the seed noise. The rise from c = 0.17 to c = 500 is
the same for the placebo (+0.177) and the real signals (+0.176), so the curve's rise is the kernel converging, not learned
predictability. The real signals add a small, roughly constant amount at high c. At z = 1e-3 (where the model nearly interpolates) real and
placebo are indistinguishable at every c (−0.7 sd at c = 500).

**Placebo at T = 60 and 120** (`python -m experiments.placebo --T 60 --shuffles 10`, `figures/placebo_T60.png`,
`placebo_T120.png`; 10 shuffles, same seed-pool scheme with the 10 real seeds), z = 1e3, mean over 159 factors:

| T | c | c = 0.17 | c = 1 | c = 16 | c_max |
|---|---|---|---|---|---|
| 60 | real | 0.277 | 0.341 | 0.345 | 0.345 |
| 60 | shuffled (sd) | 0.220 (0.088) | 0.289 (0.013) | 0.297 (0.005) | 0.297 (0.005) |
| 120 | real | 0.345 | 0.361 | 0.362 | 0.363 |
| 120 | shuffled (sd) | 0.269 (0.037) | 0.291 (0.008) | 0.293 (0.006) | 0.293 (0.006) |

The rise with c is again the same for the placebo (+0.076 at T = 60, +0.023 at T = 120) as for the real signals (+0.068, +0.018).
But at the longer windows the real signals beat the placebo by far more than at T = 12: +0.048 at T = 60 and +0.070 at T = 120
(against +0.021 at T = 12), i.e. 9–13 placebo sd at high c. So with 60–120 months of training the signals carry information
beyond the factor's own average return, even though the Sharpe does not rise with complexity beyond what the placebo shows. The placebo sits
below the trailing mean (0.309, 0.303) at both windows, while the real model at T = 120 (0.363) is 0.013 above always long
(0.351; s.e. 0.012). The sd is across shuffles (10), so "sd units" are descriptive, not a p-value.

**3.3 Exact kernel** (`python -m experiments.exact_kernel`). Exact mixture-kernel ridge (no random features) gives,
at z = 1e3, 0.343 on real and 0.321 on shuffled signals, matching the random-feature model at c = 500 (0.342,
0.321). So the "complexity" axis here is how well the sampled kernel approximates the exact one.

**3.4 Seed sensitivity at the smallest model.** At L = 2 the model has a single random (sin, cos) pair, and its
Sharpe depends heavily on that draw. In the eq. (36) set (20 seeds), the 159-factor mean at z = 1e3 ranged from −0.01 to 0.37
across seeds (sd 0.135; seeds 1000–1004 average 0.170, all 20 average 0.249; L = 4 has sd 0.07). The same seed is used for
every factor, so averaging over factors does not remove this. A first version of this analysis used 5 seeds
and reported 0.17 at c = 0.17, a "2× rise"; that was a property of those five seeds, not of the model. The
c ≥ 1 points are stable (seed sd below 0.01). With the literal eq. (37) the c = 0.17 point moved from 0.135 (20 draws) to 0.166 (100 draws).
The paper averages 100 seeds.

### 3.5 Simulator (beyond the paper; 50 simulated datasets, T = 12, 5 random-feature draws each, literal eq. 37 features)

Returns have unit variance and a small non-linear predictable part (snr = 0.05).

| Prediction | Result |
|---|---|
| (1) R² negative while Sharpe positive | Yes. R² ≈ −2.3 at c = 1 for z = 1e-3 (≈ −7 at z = 1e-4, ≈ −900 at z = 1e-8); Sharpe positive at all c. |
| (2) Sharpe dips toward 0 at c = 1 without ridge | **No at low ridge.** Sharpe c = 2/3 → 1 → 2: 0.0051 → 0.0041 → 0.0043 at z = 1e-3 (dip −0.0010, paired t = −0.6); −0.0009 (t = −0.3) at z = 1e-12. A dip appears at **large** ridge instead: 0.0033 → 0.0009 → 0.0040 at z ≥ 1e1 (t = −2.5). ‖β‖ does explode at c = 1 without ridge (4,566 at z = 1e-12), as in the theory. |
| (3) Sharpe monotone in c with enough ridge | **Beyond c ≈ 2, yes:** 0.0043 (c = 2) → 0.0235 (c = 256) at z = 1e-3 and 0.0040 → 0.0083 at z = 1e3, but not monotone through c = 1 (above). |
| Lemma 1: ‖β‖ falls in L past c = 1 | Yes (peak 8.2 at c = 1, then 5.3), plateauing instead of going to 0. |

Under the eq. (36) kernel (earlier runs, `_eq36` files) the dip appeared only at z ≤ 1e-8, and weakly (t = −2.3). Adding an unconditional drift (Sharpe 0.08 per period) did not create a
dip at z = 1e-3 (`figures/sim_theory_mu0.08.png`). Figures: `figures/sim_theory.png`, `..._z-12_-8_-4_-3.png`. So the real-data dip at c = 1
(section 1) is not reproduced by our simulator at the same z, and we do not know why.

### 4. Feature normalisation vs the paper's figures (`python -m experiments.scaling_check`, `figures/scaling_check.png`)

The paper's eq. (37) divides the features by √P, but P could be the model's width, the 7p weights, or
something else. Our main runs use √L (the model's own width), which makes the effective ridge independent
of L. Against the paper's Figure 3B (momentum signals) this choice is **not** what the figure looks like: in the
paper the z = 10 and z = 100 curves differ from z = 1e3 at high complexity (≈ 0.20 and 0.25 against 0.30 at c = 1000) and
the curves for z = 1 … 100 fall with c after a peak around c = 3–10, whereas with √L every z ≥ 10 curve coincides and is flat.
A constant normalisation S = F/√κ makes the effective ridge shrink as L grows, which produces falling curves. Mean annualised
Sharpe over 159 factors at c = 500, literal eq. (37) (paper values read off Fig. 3B at c = 1000 by eye; 10 seeds for the κ columns):

| z | paper (c = 1000) | 1/√L (main) | κ = 1 (raw) | κ = T = 12 | κ = 7p = 3500 |
|---|---|---|---|---|---|
| 1e3 | ≈ 0.30 | 0.342 | 0.331 | 0.342 | 0.342 |
| 1e2 | ≈ 0.25 | 0.342 | 0.264 | 0.333 | 0.342 |
| 1e1 | ≈ 0.20 | 0.342 | 0.218 | 0.271 | 0.342 |
| 1 | ≈ 0.17 | 0.342 | 0.202 | 0.220 | 0.341 |
| 1e-1 | ≈ 0.17 | 0.322 | 0.192 | 0.203 | 0.307 |
| 1e-2 | ≈ 0.16 | 0.249 | 0.186 | 0.192 | 0.236 |
| 1e-3 | ≈ 0.15 | 0.213 | 0.184 | 0.186 | 0.209 |

Reading (this table changed from the eq. (36) runs, where κ = 1 gave 0.030 at z = 1e-3): with the literal eq. (37) features,
**κ = 1 (unscaled features) reproduces the paper's z-dependence closely** — 0.33, 0.26, 0.22, 0.20, 0.19, 0.19, 0.18 against ≈ 0.30, 0.25,
0.20, 0.17, 0.17, 0.16, 0.15 — and so does its R² (−0.28 at z = 1e-3 against the paper ≈ −0.45; the 1/√L runs give −0.18). The ‖β‖ at z = 1e-3
(mean of 159 factors) peaks at c = 1 (κ = 1: 6.3, κ = T: 8.9) and then **falls** to 0.16 and 0.52 at c = 500, as the paper's does, whereas
under 1/√L it peaks at 9.6 (c = 2) and plateaus at 7.8, and under κ = 7p it peaks late (16.3 at c = 16). The paper's ‖β‖ falls to ≈ 1 by c = 50 for
momentum; κ = 1 is the closest. What does not depend on the normalisation: the best attainable Sharpe is ≈ 0.342 under every one of them (the
z = 1e3 plateau, κ = 1 reaches 0.331), so the "no edge over always long (0.353)" and the placebo conclusions do not change; what changes is how the
curves depend on z and c. **We have not rerun the whole pipeline with κ = 1**, so the main figures use 1/√L; that, and checking κ = 1
on the Welch–Goyal panel, is the obvious next step. The paper does not state its normalisation, so κ = 1 is a candidate, not a finding.

### 5. Kernel check, and eq. 36 vs eq. 37 literal (`python -m experiments.kernel_check`)

**The kernel implementation is correct.** Random features reproduce the exact mixture kernel for both conventions
to within sampling error (max |error| 0.011 and 0.014 against 1/√4000 = 0.016; `figures/kernel_check.log`), and
`tests/` checks the exact-kernel ridge against a many-feature model.

**What the kernel looks like on our signals** (standardised, pairwise distance median 1.4). Mean off-diagonal K per bandwidth:

| γ | 0.1 | 0.5 | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|---|
| eq. 36 (frequencies ω/√γ) | 0.08 | 0.27 | 0.41 | 0.56 | 0.70 | 0.81 | 0.89 |
| eq. 37 literal (frequencies γω; main) | 0.98 | 0.70 | 0.41 | 0.16 | 0.05 | 0.01 | 0.00 |

**This explains the placebo.** At infinite ridge the position Σ w_s y_s (w_s ∝ K(x_t, x_s)) uses almost all past returns with almost
uniform weights, under both conventions: on 20 factors it uses 11.1 of the 12 past returns for real signals and 11.0 for shuffled ones
(eq. 37; largest weight 0.13–0.14 against a uniform 0.08), and the position has correlation 0.966 (real) and 0.963 (shuffled) with the trailing mean
(eq. 36: 0.956 and 0.959). So the large-z strategy is essentially a trailing mean whatever the signals are.

**Why eq. 37 is the main convention.** The paper's own text says that "when γ is close to zero, (8)–(9) is approximately equivalent to a linear
model". That holds for frequencies γω (small γ = low frequency = nearly linear, row 2 above) and not for ω/√γ. We first followed eq. (36)
and changed to the literal eq. (37) after the kernel check, then reran everything. Same 159 factors, T = 12, momentum signals:

| | eq. 36 (20 draws) | eq. 37 literal (100 draws; main) | paper (by eye) |
|---|---|---|---|
| Large-z Sharpe, c = 0.17 → 500 | 0.249 → 0.347 (1.4×) | 0.166 → 0.342 (2.1×) | ≈ 0.1 → 0.30 (≈ 3×) |
| z = 1e-3 Sharpe: minimum at; value at c = 500 | c = 2 (0.105); 0.142 | **c = 1** (0.093); 0.213 | dip near c = 1 (≈ 0.07); ≈ 0.15 |
| ‖β‖ at z = 1e-3 | 2.0 → 11.0, plateau | 2.1 → 9.6 (c = 2), then falls to 7.8 | falls past c ≈ 1–2 |
| ER / Vol per month at z = 1e-3, c = 500 | 0.041 / 0.99 | 0.039 / 0.63 | ≈ 0.04 / ≈ 0.8–0.9 |
| OOS R² at z = 1e-3, c = 500 | −0.51 | −0.18 | ≈ −0.45 |
| Model minus always long | −0.006 (s.e. 0.017) | −0.011 (s.e. 0.017) | not reported |
| T = 60 / 120, large z, c_min → c_max | 0.340 → 0.352 / 0.365 → 0.370 | 0.277 → 0.345 / 0.345 → 0.363 | monotone rise |
| Placebo, real − placebo at c_max (T = 12 / 60 / 120) | +0.027 / +0.057 / +0.080 | +0.021 / +0.048 / +0.070 | not reported |

The literal reading reproduces the dip at c = 1, the gain of about 2–3×, the direction of ‖β‖ and the Fig. 5 rise better; it matches
Vol and R² worse. The conclusions on always long, the placebo and the exact kernel are the same under both conventions.

### Side by side with the paper

| | Paper (Fig. 3/5) | Here: momentum signals | Here: Welch–Goyal signals |
|---|---|---|---|
| Sharpe, low complexity, large z | ≈ 0.1 (B), ≈ 0.11 (A) | 0.166 (c = 0.17) | 0.117 |
| Sharpe, high complexity, large z | ≈ 0.30 (B), ≈ 0.36 (A) | 0.342 | 0.368 |
| Gain from low to high c | "a factor of three" (text, p. 26) | 2.1× | 3.1× |
| No-ridge curve (z = 1e-3) | dip near c = 1 to ≈ 0.07, then ≈ 0.15 (B); ≈ 0.07 → ≈ 0.32 (A) | 0.121, min 0.093 at c = 1, 0.213 | 0.079, min 0.079 at c = 1, 0.305 |
| z = 10 and 100 curves | differ from z = 1e3 at high c and fall with c (B) | coincide with z = 1e3 (scaling 1/√L); κ = 1 reproduces them (section 4) | coincide |
| T = 60, 120 (Fig. 5) | monotone rise | rise, small (+0.068, +0.018) | +0.069, +0.030 |
| ‖β‖ (Fig. 1) | falls with c past c = 1 | peaks at c = 2, falls to a plateau of 7.8 | plateau of 6.1 |
| ER at low z (Fig. 2) | ≈ 0.04 | 0.039 | 0.038 |

The paper's values are read off its figures by eye, and the paper does not say whether Sharpe ratios are annualised (I annualise;
its Fig. 2 values for ER and Vol are consistent with annualised Sharpe ratios of about the size in its Fig. 3). Note that the paper's ≈ 0.30–0.36 at
high complexity is close to the mean Sharpe of the 159 factors in its own Table 1 (0.31) and to always long here (0.349–0.353).

## What didn't work, or didn't match

1. **The size of the effect for momentum signals.** 2.1× gain in Sharpe at T = 12, not ≈ 3×, and +0.018 to +0.068 at
   T = 60 and 120. With Welch–Goyal signals the 3.1× gain matches. We do not know whether the momentum gap is our signal construction,
   our feature normalisation (section 4), or the sample.
2. **No demonstrated timing edge.** The best model (0.342 for momentum, 0.368 for Welch–Goyal) is −0.011 and +0.019 from always long
   (0.353 and 0.349), s.e. 0.017–0.018, so within noise. This does not show the model *loses*; it shows the paper's evidence does
   not include a baseline that would let one say it wins.
3. **The z-dependence of the curves and ‖β‖.** Our curves for z ≥ 10 coincide and ‖β‖ plateaus; the paper's differ and fall. Section 4 finds
   an unscaled-feature version (κ = 1) that reproduces them under eq. (37), but we did not rerun the pipeline with it, so the main
   figures do not have this. **Partly resolved.**
4. **The double-ascent dip.** On real data the z = 1e-3 minimum is at c = 1, as in the paper. In our simulator, the dip at c = 1 appears
   only at large ridge (t = −2.5) and not at z ≤ 1e-3 (t = −0.6), so the theory's prediction (2) is not reproduced there, whereas ‖β‖ blows up at
   c = 1 as predicted. **Unresolved.**
5. **The rise with c is mostly there without any signal information** (placebo, section 3.2).
6. **Our own first results were partly wrong.** The first 159-factor run used 5 feature seeds and the placebo a
   single seed, so the smallest-model points compared different random draws and we reported a "2× rise" and a
   "+14 sd" real-vs-placebo gap that were seed artifacts (corrected; section 3.4).
7. **We changed the main kernel convention after seeing results.** We started with eq. (36), compared with the paper's figures, and switched to the
   literal eq. (37) because it fits better (section 5). That is a choice informed by the target, not a blind replication; the eq. (36)
   results are kept as a sensitivity set.
8. **Why the placebo matches** is tested (section 5): at infinite ridge the kernel position is a near-uniform average of the last T returns
   (correlation 0.96 with the trailing mean, real or shuffled). Still untested: (a) which bandwidths drive it, (b) why very small models
   do poorly (I suspect random-sign kernel weights at L = 2).
9. **Not done:** see "What we did not do". All conclusions are limited to US equity factors.

## Where the paper is ambiguous, and what we chose

| Item | Choice |
|---|---|
| **Feature scaling** | Paper: S = P^{-1/2} h. We scale by the width L of the model being fitted (G = G_raw / L). With a fixed 1/√P_total, "z = 1e-3" is not near-ridgeless at small L and R² does not explode at c = 1. This is an **inference**, not stated in the paper, and section 4 shows it does not reproduce the paper's z = 10 / 100 curves; an unscaled version (κ = 1) does. **Partly resolved.** |
| **Eq. (36) vs (37)** | Taken literally, eq. (37) (frequencies γ·ω, ω ~ N(0, I)) gives a kernel exp(−γ²‖Δx‖²/2), not eq. (36)'s exp(−‖Δx‖²/(2γ)). Main runs use the literal eq. (37) (`convention="paper"`); eq. (36) is `convention="kernel"` (files `_eq36`). Section 5 shows why. |
| **Number of features** | Each of the P = 7p weights gives a (sin, cos) pair, so 2P columns; L counts columns. |
| **Eq. (15)** | The sum over τ = 1…t−37 (divided by t−37) read literally is the mean over observations 37…t−1: it drops the *first* 36 observations. We use an expanding RMS of all data strictly before t with ≥ 36 observations; for Mom the two differ by ≤ 15% (median 1%) after 120 observations. |
| **Welch–Goyal predictors** | The paper's footnote lists dfy, infl, svar, de, lty, tms, tbl, dfr, dp, dy, ltr, ep, b/m, ntis plus one lag of the market return. Built from Goyal's monthly spreadsheet: dfy = BAA − AAA, tms = lty − tbl, dfr = corpr − ltr, dp = log D12 − log Index, dy = log D12 − log lagged Index, ep = log E12 − log Index, de = log D12 − log E12. `infl` is lagged one month (publication delay), "market return" is the CRSP S&P value-weighted return of month t, signals are standardised by an expanding RMS over the full 1926-12 history and clipped at ±5, and the sample stops at 2020-12 as in the paper. These are our choices; the downloaded vintage is not the paper's. |
| **R² denominator** | Eq. (41) is garbled in the PDF; the text after eq. (52) implies the uncentred 1 − MSE/E[R²]. We centre on the sample mean. When π ≈ 0 (large z) centred gives ≈ −0.02 and uncentred 0.000. |
| **Annualisation** | Not stated for Fig. 3. We annualise monthly Sharpe by √12. |
| **Timing** | Signals at t, target y[t] = r[t+1]/σ_t, trained on pairs t−T…t−1 (all known at t). `tests/` checks that changing future data does not change a past forecast. |

## Tests

`pytest -q` runs 13 tests: the dual-form ridge equals the paper's eq. (9) computed literally (both
scalings); no look-ahead; random features approximate the mixture kernel (both conventions); the ridgeless solution
interpolates and has minimum norm (Lemma 1); metrics; exact kernel ridge equals a many-feature
random-feature model; the 159-series universe is clean and tracks the paper's Table 1; the placebo and
real runs share one seed pool; the Welch–Goyal predictors follow their definitions, are lagged correctly and end in 2020.

## Layout

```
voc/            reusable modules (features, ridge, kernel, metrics, preprocess, simulate)
experiments/    curves (the replication: paper Figs 1-5 on 159 factors), common (shared helpers),
                check_table1, report_numbers, and beyond the paper: sim_theory, placebo, exact_kernel, kernel_check, scaling_check;
                plot_paper_axes and plot_scaling_slide redraw saved results on the paper's axes (used by the slides and notebook)
data/           fetch_data.py + ff_factors_monthly.csv, jkp_factors_usa_monthly.csv, welch_goyal_monthly.csv, paper_table1.csv (committed)
figures/        generated figures, .npz result arrays, full-run logs, report_numbers*.log
notebooks/      virtue_of_complexity.ipynb: walkthrough of the implementation and all results
                (live code checks; heavy results are re-plotted from figures/*.npz)
slides/         presentation (LaTeX/Beamer source and PDF, the method pages, paper figure crops)
paper/          the paper under replication (SSRN 4166368) and its companion (NBER 30217)
tests/          unit tests
run_all.py      single entry point
AI_USAGE.md     AI usage disclosure
```
