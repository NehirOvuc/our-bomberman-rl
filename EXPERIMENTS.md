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
| v7 | **decision forest instead of a line** | 1.260 | 7.68 | 3.38 | **it bombs** |
| v8 | **+ 5-step reward window (Nehir)** | **2.065** | 6.96 | **13.52** | **biggest jump** |
| v9 | **+ symmetry augmentation restored (Daniel)** | **2.466** | 8.63 | **19.01** | **best so far** |

Alone on the board: v1–v5 scored **0.000**. v7 scores **0.155**, v9 scores **1.217**.

*(v6 was v7 with a buffer bug that flattered it to 1.902 — excluded.)*

**v8 is the biggest single result and it was Nehir's.** A bomb goes off six
steps after you drop it (`BOMB_TIMER` 4 + `EXPLOSION_TIMER` 2), but the model
was only ever shown the very next step — so dropping a bomb looked like it
achieved nothing. Widening the window to five steps: **four times the crates
from fewer bombs**, and the invalid-move rate more than halved on its own with
no change to the reward table. It also beat the gain from switching to the
forest (+0.805 against +0.788), which is why "the model type was the binding
constraint" has been withdrawn from the report. `train.py` turned out never to
have read `n_step` on the replay path, so every arm before v8 ran at 1.

**v9 is the second biggest, and it was Daniel's.** Merging PR #1 he noticed that
Nehir's symmetry augmentation — the trick that turns every board into eight by
flipping and rotating it — had been silently dropped when the training loop was
rewritten. Putting it back was worth **+0.4** on its own: 2.065 → 2.466, and the
two confidence intervals do not overlap. The agent bombs *more* (6.96 → 8.63)
and kills itself *less* (0.207 → 0.123 suicides per round), which is what eight
times the practice at escaping your own bomb should buy. Alone on the board it
went from 0.155 to **1.217**.

One number went the wrong way: the share of moves that are illegal rose from
6.7% to 9.4%. That is expected and is written up in the report. An illegal move
leaves you exactly where you were, so the model still keeps almost all the value
it had, and the −0.1 we charge for it has to beat the model's own error — which
grows as the scores grow. The penalty did not get weaker; the numbers it is
competing against got bigger.

Note for anyone comparing rows: **v1–v8 were all trained without augmentation
and v9 with it.** The eight-arm table above is still a fair comparison within
itself, but v9 is not a ninth arm in the same series — it changes one thing
against v8 and two against v7.

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
3. **We concluded the fix was to make the model forget old data — before the
   results were in.** Both versions of that were then measured and both made the
   agent clearly worse. Withdrawn.

---

## For Daniel

**All taken into PR #1 (`e4ddd37`) — this section is history now.** What was here:
`model_b.py` (the forest, a new file so `model.py` was untouched), the
`callbacks.py` resume fix, and `train.py`'s cross-stage replay buffer plus the
n-step targets that the setting had never actually been wired to.

Both items that were open are fixed on his branch: Q-value ties now break
randomly instead of always picking UP, and the agent has its own seeded RNG
(`TACO_SEED`) so a run is reproducible. He also restored the symmetry
augmentation that the rewrite had dropped — that is v9, above.

**One thing to know before running the forest again:** augmentation makes one
refit cost **75 s instead of 6.8 s** on a full buffer. Eight times the data is
eighteen times the tree-fitting cost, because that scales superlinearly — Model
A's entire refit is 10.6 s by comparison. Peak memory 0.7 → 3.0 GB, saved model
9.3 → 19.6 MB, a full curriculum about 4 hours. Step time is unaffected at
7.4 ms against a 500 ms budget.

## For Nehir

- PR #3 is reviewed and should be merged.
- Your curriculum is what every arm since v2c trains on.
- Your interaction feature vs the forest is the experiment that now decides
  Section 6. It needs a matched scenario before either result means anything —
  and the baseline it has to beat is now **v9's 2.466**, not v8's 2.065, so both
  linear arms have to be trained with augmentation on.

---

## Running it

```bash
.venv/bin/python -m pytest tests -q                    # 161 tests

# evaluate any version (appends a row to experiments/experiment-log.csv)
TACO_MODEL=b .venv/bin/python tools/evaluate.py --agent taco_kebab_agent \
    --version <label> --lineup tournament|solo --note "what this tested"

# train one curriculum stage
TACO_MODEL=b TACO_FRESH=1 .venv/bin/python main.py play --my-agent taco_kebab_agent \
    --train 1 --no-gui --n-rounds 800 --seed 1 --scenario coin-heaven
```

Full numbers: `experiments/experiment-log.csv`.
