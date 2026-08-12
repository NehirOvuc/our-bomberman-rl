# Bomberman RL — Team Plan (3 members, target finish 31.08.2026)

**Official deadlines:** submission test 17.09 · agent code 21.09 21:00 · report 28.09 21:00
**Our target:** code + report done **31.08**, submitted first week of September.
**Today:** 11.08.2026 → three working weeks.

---

## 0. The constraint that drives everything

> "do **not** split labor such that every team member works on their separate model — we attach great importance to real teamwork!"

So we do **not** split by model. We split by **pipeline layer**. Both models are built on the *same* features, the *same* rewards, and the *same* evaluation harness. Every member's work is used by both models, so nobody owns a model and nobody's contribution is isolated.

| Warning in the brief | How the plan handles it |
|---|---|
| No multiprocessing in the *final* agent (training is fine) | C's packaging script greps for `multiprocessing`/`joblib` before freezing. |
| Final agent runs on **CPU**, 0.5 s/step, 8 GB RAM | A budgets feature extraction; C runs a per-step latency benchmark weekly. |
| Framework changes won't exist in the tournament | Agent logic lives **only** in `agent_code/<our_agent>/`. Modified `settings.py`/scenarios live on a `training-env` branch, never in the zip. |
| Auxiliary rewards absent in official games | All evaluation runs with rewards **off**; shaping is state-dependent (potential-based) where possible. |
| Model must *learn* from features | No feature may return "the best action". A + C code-review every feature against this. |
| `settings.py` may change until 14.09 | Both models retrain in hours, not days. One person stays on call in September. |
| Grade is mostly **systematic method**, not tournament rank | C's experiment log is a primary deliverable, not an afterthought. |
| Report marks main author per section | Section ownership assigned below, matched to actual work. |
| **"If you do not refine AI drafts in your own style, you will fail the examination."** | See §1, Claude Code working agreement. This is a hard rule, not advice. |

---

## 1. Claude Code working agreement

Everyone is using Claude Code, which is what makes a three-week schedule realistic. It also introduces the single biggest risk to the grade: code and prose that nobody on the team can defend. Four rules:

1. **Interface contract before any generation.** Day one, the three of us agree the exact signatures of `state_to_features`, `Model.predict_q/update/save/load`, and `reward_from_events`. Everyone generates against the agreed stubs. Without this, three people produce three incompatible codebases in an afternoon.
2. **The explain test.** Before any PR merges, the author walks the reviewer through it verbally, without the assistant. If you can't explain why a line is there, it doesn't merge. Tutors do ask.
3. **Never generate the report.** Use the assistant to structure, critique and tighten. Write the sentences yourself. Your §6 has to describe experiments *you* ran and decisions *you* made — that content doesn't exist in a model.
4. **Log the loop, not the output.** Every experiment gets a row in the log with your hypothesis and your decision. That reasoning is the thing being graded, and it's the part the assistant cannot supply.

What Claude Code does **not** accelerate, so don't plan as if it does: training wall-clock, RL convergence, and the run→measure→think loop on bomb escape.

---

## 2. Models

- **Model A — Q-learning with linear function approximation.** Q(s,a) = βₐ·φ(s), one weight vector per action, fitted by **ridge-regularised least squares** on n-step TD targets. Converges in hours, microseconds per step on CPU. *Tournament fallback — this one must work.*
- **Model B — Fitted Q-iteration with a regression forest.** Identical φ(s), non-linear Q-function (sklearn random forest or gradient boosting), periodically re-fit on the replay buffer.

Only the function approximator differs. That makes §6 a genuine controlled experiment — *linear vs non-linear function approximation on identical inputs* — rather than "we tried two unrelated things." No torch dependency, no GPU, no convergence gamble, and a settings change in September costs an afternoon.

**Optional third model (stretch, decide 24.08):** a small CNN DQN on a board tensor. Only start it if Tasks 1–3 are green and both models are trained. It is never allowed to become the critical path.

### Mapping to the lecture

| Design decision | Lecture |
|---|---|
| Hand-crafted feature vector φ(s) | Features & response; augmented features for non-linear boundaries |
| Q(s,a) = βₐ·φ(s) fitted by least squares | OLS: normal equations, pseudo-inverse, Cholesky/QR |
| Ridge penalty on βₐ | Bias–variance trade-off, ridge regression |
| Forest Q-function | Non-linear regression: regression trees & forests |
| Q-learning, TD targets, ε-greedy | Model-free reinforcement learning |
| Held-out seeds for evaluation | Training vs test set, cross-validation |
| Symmetry-based data augmentation | Training tricks |
| *(if the stretch DQN happens)* | Neural networks, backprop, ConvNets, Deep Q-Learning |

Report §2 (Background) can be written more or less straight off this table.

---

## 3. Role split

### Member A — Environment, Features & State Representation
- Repo setup, branch policy, `requirements.txt` (numpy/scipy/sklearn only).
- `features.py` — target ~25 dims: situational awareness (wall/crate/bomb in each of 4 directions), BFS direction to nearest coin / nearest crate / nearest opponent, danger map (in a blast radius? in how many steps?), escape-route availability, bomb-usefulness indicator.
- `symmetry.py` — the 8 dihedral transforms plus the matching action permutation, for data augmentation.
- CPU latency micro-benchmark for feature extraction.
- Training scenarios in `settings.py` on the `training-env` branch.

**Report:** §2 (state representation, function approximation), §4 (feature engineering, symmetries).

### Member B — Learning Algorithms & Models
- `model.py` — one interface, both models: `predict_q(features) -> np.array(6)`, `update(batch)`, `save()`, `load()`.
- Model A: ridge-linear Q per action, n-step TD targets, ε-greedy with decay and a floor.
- Model B: fitted Q-iteration, forest re-fit on the buffer at a fixed cadence.
- `train.py` callbacks (`setup_training`, `game_events_occurred`, `end_of_round`), replay buffer.
- Grid search over γ, learning rate, ε schedule, ridge λ, n-step, forest depth/estimators. Multiprocessing allowed **for training only**.

**Report:** §2 (RL algorithms), §4 (model design), §5 (training).

### Member C — Rewards, Evaluation & Operations
- `rewards.py` — the 16 built-in events plus custom ones (`MOVED_TOWARDS_COIN`, `MOVED_AWAY_FROM_COIN`, `ESCAPED_BLAST`, `STAYED_IN_BLAST`, `WAITED_IN_DANGER`, `DROPPED_USELESS_BOMB`, `BOMB_NEXT_TO_CRATE`, `TRAPPED_SELF`). Balanced ± pairs so there's no back-and-forth farming exploit.
- `evaluate.py` — N episodes headless, `--train 0`, held-out seeds, fixed opponent line-ups. Metrics: mean score, kills, deaths, **suicides**, coins, invalid-action rate, survival steps, win rate vs `rule_based_agent`, with confidence intervals.
- **The experiment log** — one row per experiment: hypothesis → change → config → result → decision. Backbone of §6.
- Plots: training curves, model comparison, ablation.
- Ops: Docker submission test, relative-path audit, no-multiprocessing check, latency check, final zip, MaMPF display names, team-name registration, repo hygiene (report **not** in the repo).

**Report:** §1, §3, §6 (largest section), §7.

### Shared
- **Daily 15-minute standup** (three-week schedule; weekly is too slow).
- **Every PR reviewed by one other member**, with the explain test. No direct pushes to `main`.
- **Rotating pairing:** week 1 A+B, week 2 B+C, week 3 A+C.
- §3 of the report can honestly state the split was by *component*, not by model. Say so explicitly — it's what they're asking for.

---

## 4. Three-week timeline

### Week 1 — 11–17.08 · Build the pipeline in parallel
Day one, together: clone the repo, run `python main.py play`, and agree the interface contract. Then all three work simultaneously against stubs — no hand-off chain.

| | A | B | C |
|---|---|---|---|
| | Feature vector v1 + BFS utilities + unit tests | Ridge-linear Q + `train.py` wiring against a stub feature fn | Eval harness + reward table + baseline numbers for the provided agents |

**Gate (17.08):** agent runs end to end without crashing; `evaluate.py` produces the metric table; baselines recorded.
**Also this week:** register the team name; fix MaMPF display names; draft report §1 and §3.

### Week 2 — 18–24.08 · Tasks 1–2, both models live
| | A | B | C |
|---|---|---|---|
| | **Feature freeze 18.08**; symmetry augmentation | Model A converges on coin-heaven, then crates; Model B (forest) online | Escape/danger events; suicide rate as the headline metric; first training curves |

**Gate (24.08):** Task 1 solved; Task 2 agent survives reliably — **suicide rate is the number that matters**, not score. Both models trained and compared.
**Also:** draft §2 and §4 (methods are known — don't wait for results). Decide on the stretch DQN.

### Week 3 — 25–31.08 · Tasks 3–4, experiments, report
| | A | B | C |
|---|---|---|---|
| | Feature ablation study | Hyperparameter grid; longest training runs | Full experiment set, ablation plots, model comparison, Docker test |

**Gate (31.08):** beats `rule_based_agent` on held-out seeds; experiment log complete; §5 and §6 written; §7 written; full read-through where each member rewrites the others' drafts in the team's voice.

### Buffer — 01–07.09
Docker submission test upload, final long training run against the frozen agent, submit code and report. Then one person stays on call until 14.09 in case `settings.py` changes.

---

## 5. Risks

| Risk | Mitigation |
|---|---|
| **Task 2 bomb escape eats the schedule** (the usual failure) | Highest-priority work in week 2. If suicide rate is still bad on 24.08, cut the ablation study, not the escape work. |
| Three incompatible codebases from parallel generation | Interface contract on day one; daily standup; PR review. |
| Code nobody can explain | The explain test blocks the merge. |
| `settings.py` changes on 14.09 | Both models retrain in an afternoon. Keep the pipeline runnable. |
| Reward shaping drives a bad local optimum | Balanced ± pairs; state-dependent shaping; evaluate with rewards off. |
| Everyone writes §6 at the last minute | §1–§4 drafted in weeks 1–2 before results exist. |

**Realistic hours:** 15–20 h per person per week. Below that, this is a September plan, not an August one.