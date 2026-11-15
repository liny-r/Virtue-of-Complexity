# AI usage note

**Tool:** Claude Code (Anthropic's CLI coding agent, model Claude Sonnet 5.5), used interactively in VS Code.

## What the agent did
- Read the assignment PDF and the paper, and proposed the project scope and plan.
- Wrote the code in `voc/`, `experiments/`, `data/fetch_data.py`, `tests/` and `run_all.py`
  from equations in the paper, ran every experiment, and drafted `README.md` from the outputs.
- Proposed the follow-up analyses (ridgeless check, drift test, placebo test, exact-kernel check).

## What was checked, and how
- `tests/test_core.py` (7 tests) checks the implementation against the paper's equations:
  the dual-form ridge equals eq. (9) computed literally; a forecast at time t does not change
  when later data are altered (no look-ahead); the random features approximate the mixture
  kernel of eq. (36); the ridgeless fit is the minimum-norm interpolator (Lemma 1); the exact
  kernel ridge matches a many-feature random-feature model.
- The downloaded factor data were compared with the paper's Table 1 (mean, volatility, Sharpe of
  the six factors); they agree to within data-revision differences (see README).
- Where the paper is ambiguous (eqs. 15, 36/37, the feature scaling, annualisation) the choice made
  is stated in the README and in the module docstrings; these are interpretations, not facts
  taken from the paper.

## What I verified myself
<!-- TODO (student): replace this paragraph with your own account, e.g. which derivations you
re-did by hand (the dual form, the 1/L scaling, the kernel/feature identity in eq. 37), which
lines of code you traced, and anything you changed. Do not submit with this placeholder. -->

## Not AI-generated
The decision about what the paper's one real idea is, what to cut, and the critique in the talk
are the presenter's own.
