# AI usage note

**Tool:** Claude Code (Anthropic's CLI agent, Claude Sonnet 5.5), used interactively in VS Code, mainly by Yueqi Lin.

## What the agent did
- **Code and results:** wrote `voc/`, `experiments/`, `data/fetch_data.py`, `tests/` and `run_all.py` from the paper's equations,
  ran every experiment, and drafted `README.md` from the saved outputs. The follow-up checks beyond the paper
  (simulator, placebo, exact kernel, feature-scaling check) were proposed by the agent.
- **Data:** found and downloaded the JKP factor data, built the Welch-Goyal predictors from Goyal's public spreadsheet,
  and transcribed the paper's Table 1 from the PDF into `data/paper_table1.csv`.
- **Notebook, figures, slides:** built the notebook, the paper-axis figures and the Beamer slides for the replication
  part of the talk, and drafted speaker scripts; the presenters then edited them.
