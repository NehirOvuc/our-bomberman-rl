# What we tried, why, and what happened

Everything Ege ran between 28.08 and 02.09, on one branch so you don't have to
read six. Numbers are the framework's own scoring — our rewards switched off —
on ten held-out seeds (900–909) never used in training, 100 rounds each.

> ⚠️ **This branch must never be merged into `master`.** It carries `settings.py`
> and `training_scenarios.py` from `training-env` so the curriculum can run.
> Take the agent code from it; leave those two files behind.

---

## The short version

The agent would not drop bombs. Every coin in `classic` starts inside a crate,
so an agent that won't bomb can only collect what opponents accidentally
uncover — which is why it scored **exactly 0.000 alone on the board**, on all
ten seeds, for seven straight versions.

Five things were tried and none worked. The sixth did: **changing the model
from a straight line to a decision forest.** It bombs, and it is the first
version to score anything at all alone.

---

## Every arm, in order

| # | what changed | score | bombs/rd | crates/rd | verdict |
|---|---|---|---|---|---|
| v1 | the reward table from the contract | 0.779 | 0.00 | 0.00 | control |
| v2 | + 4 bomb/danger rewards | 0.764 | 0.00 | 0.00 | no effect |
| v2c | + Nehir's staged curriculum | **0.899** | 0.00 | 0.00 | best linear |
| v3 | + model forgets old data | 0.176 | 0.11 | 0.02 | much worse |
| v4 | + replay buffer | 0.211 | 0.01 | 0.00 | **measured wrong, see below** |
| v5 | v4 with the buffer bug fixed | 0.472 | 0.00 | 0.00 | worse than v2c |
| v7 | **decision forest instead of a line** | **1.260** | **7.68** | **3.38** | **it bombs** |

Alone on the board: v1–v5 scored **0.000**. v7 scores **0.155**.

*(v6 was v7 with a buffer bug that flattered it to 1.902 — excluded. v8, the
forest with a longer reward window, is running now.)*

---

## Why each change was made

- **v2 — more rewards.** The obvious first move: pay it to bomb well. It didn't
  work, and the reason is the useful part. At its escape skill a bomb was worth
  about −2.0 to it; a +0.3 bonus closes 13% of that. Anything big enough to
  close it would be paying it to die.
- **v2c — Nehir's curriculum.** Train on easy maps first. It's the only thing
  that moved survival per bomb (38% → 45%) — but 45% still isn't enough to make
  bombing worth it.
- **v3/v4 — make the model forget.** Nehir's PR #1 point: the fit accumulates
  and never lets old data fade. Both versions made the score much worse, because
  fresher targets inflate the Q-values and the agent starts following noise.
- **v7 — a different model.** Five arms had changed rewards, staging and
  fitting. The one thing never changed was the model itself. Same features, same
  rewards, same curriculum, same seeds — only a forest instead of a line.

---

## Three things we got wrong and fixed

1. **The replay buffer never worked as described.** Each curriculum stage runs
   as a separate process and the buffer was rebuilt empty every time. So it never
   spanned more than one stage, and enlarging it couldn't help. Fixing it moved
   that arm 0.211 → 0.472. The conclusion "refit-from-buffer is ruled out" was
   wrong and is withdrawn.
2. **"A straight line can't express 'crate AND escape'" is false.** It ranks that
   case top perfectly well. What it can't do is get the *near-misses* right, and
   Nehir has since shown that adding the two features multiplied together fixes
   it — so the gap was in the features, not the model type. **That comparison is
   still open.**
3. **Ege told Daniel to build the forgetting/replay mechanism before the results
   were in.** The results say don't. Withdrawn.

---

## For Daniel

Prototypes on this branch, in your files. Yours to take, rewrite or reject.

- `model_b.py` — the forest. New file, `model.py` untouched.
- `callbacks.py` — training now resumes from saved weights (it always started
  from zero, which made the curriculum impossible to run). `TACO_MODEL=b` and
  `TACO_NSTEP=5` switch arms.
- `train.py` — replay buffer survives between stages; n-step targets actually
  wired up (the setting existed but was connected to nothing).
- **Still open:** an untrained model returns six zeros and picks UP, 46% invalid
  moves early on. Needs random tie-breaking. Exploration also uses numpy's global
  RNG, so runs aren't reproducible under `--seed`.

## For Nehir

- PR #3 is reviewed and should be merged.
- Your curriculum is what every arm since v2c trains on.
- Your interaction feature vs the forest is the experiment that now decides
  Section 6. It needs a matched scenario before either result means anything.

---

## Running it

```bash
.venv/bin/python -m pytest tests -q                    # 132 tests

# evaluate any version (appends a row to experiments/experiment-log.csv)
TACO_MODEL=b .venv/bin/python tools/evaluate.py --agent taco_kebab_agent \
    --version <label> --lineup tournament|solo --note "what this tested"

# train one curriculum stage
TACO_MODEL=b TACO_FRESH=1 .venv/bin/python main.py play --my-agent taco_kebab_agent \
    --train 1 --no-gui --n-rounds 800 --seed 1 --scenario coin-heaven
```

Full numbers: `experiments/experiment-log.csv`.
