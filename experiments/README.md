# Experiments

`experiment-log.csv` — one row per evaluation, appended automatically by
`tools/evaluate.py`. Do not edit it by hand; re-run the evaluation instead, so
that every row has a git commit next to it and can be reproduced.

Columns: date, commit, agent, version, scenario, lineup, seeds, rounds,
score_mean, score_ci, coins, kills, suicides, bombs, crates, invalid_rate,
steps_survived, seeds_won, note.

Every figure is a **per-round mean**, and every interval is a 95% t-interval
over the per-seed means, so n is the number of seeds and not the number of
rounds. See the module docstring of `tools/evaluate.py` for why the framework's
statistics file makes a per-round distribution unavailable.

This directory is tracked on purpose. `docs/` is local working notes and is
gitignored; the experiment log is shared evidence and is the backbone of the
report's Experiments section.
