# AI usage note

**Tool:** Claude Code (Anthropic's CLI coding agent, model Claude Sonnet 5.5), used interactively in VS Code.

## What the agent did
- Read the assignment PDF and the paper, and proposed the project scope and plan.
- Wrote the code in `voc/`, `experiments/`, `data/fetch_data.py`, `tests/` and `run_all.py`
  from equations in the paper, ran every experiment, and drafted `README.md` from the outputs.
- Proposed the follow-up analyses (ridgeless check, drift test, placebo test, exact-kernel check).
- Found and downloaded the JKP factor data (jkpfactors.com), transcribed the paper's Table 1 from
  the PDF into `data/paper_table1.csv`, and wrote `check_table1.py` to compare them.
- Restructured the repo at the user's request to replicate only what the paper reports (its 159-factor
  universe, Figs 1-5), keeping the simulator and extra checks as clearly labelled "beyond the paper" work,
  and parallelised the per-factor loops (results identical to the serial run).
- Audited the code against the paper's equations and Figure 3 (feature normalisation, eq. 15, the
  R² denominator), and ran `scaling_check.py`. This audit also caught an error in the agent's own
  earlier results: the placebo and real runs had used different random-feature seeds, which
  distorted the smallest-model results. The runs were redone and the README corrected.
- At the user's request, added the paper's first study (15 Welch-Goyal predictors, built from Amit Goyal's
  public spreadsheet), compared the kernel conventions of eqs. (36)/(37) with `kernel_check.py`, switched the
  main runs to the literal eq. (37) and reran everything (100 feature draws at T = 12), added
  `report_numbers.py` (prints every number quoted in the README from the saved results), and rewrote the
  README and the notebook around the new results. The agent made the eq. (36) -> (37) switch after seeing which
  version fits the paper's figures better; this is disclosed in the README ("What didn't work" item 7).

## What was checked, and how
- `tests/test_core.py` and `tests/test_data.py` (13 tests) check the implementation against the paper's equations:
  the dual-form ridge equals eq. (9) computed literally; a forecast at time t does not change
  when later data are altered (no look-ahead); the random features approximate the mixture
  kernel (both conventions); the ridgeless fit is the minimum-norm interpolator (Lemma 1); the exact
  kernel ridge matches a many-feature random-feature model; the Welch-Goyal predictors follow their
  definitions, are lagged as intended and end in 2020.
- The downloaded factor data were compared with the paper's Table 1 for all 158 factors that can be
  matched by name (mean, volatility, Sharpe; correlations above 0.99), and the JKP construction was
  compared with Jensen, Kelly and Pedersen (SSRN 3774514). The agent read the PDFs and typed the
  comparison; the paper's Table 1 transcription in `data/paper_table1.csv` was parsed from PDF text
  and has not been checked by hand row by row.
- Paper values quoted from figures (Figure 3B levels) are read off by eye and are approximate.
- Where the paper is ambiguous (eqs. 15, 36/37, the feature scaling, annualisation) the choice made
  is stated in the README and in the module docstrings; these are interpretations, not facts
  taken from the paper.


## Student's own verification (to be written by the student before submission)

_Not filled in by the agent: describe in your own words what you re-derived by hand, which results you re-ran
or spot-checked, what you changed or rejected, and which parts you can defend in the session._
