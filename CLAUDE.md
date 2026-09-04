# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A course framework (`bomberman_rl`) for training an RL agent to play Bomberman, plus this team's
agent implementation. The framework code (`main.py`, `environment.py`, `agents.py`, `settings.py`,
`items.py`, `events.py`, `replay.py`, `fallbacks.py`) is provided by the course and generally should
not be modified — it will be swapped for a fresh copy at tournament time. **All of our own code goes
in `agent_code/<agent_name>/`.**

## Language convention

All generated code must be in English, with no exceptions: function and variable names, comments,
docstrings, log messages, and commit messages.

## Commands

Run from the repo root.

```bash
# Setup: install dependencies
pip install -r requirements.txt

# Play interactively with GUI (arrow keys + space to bomb, enter to wait)
python main.py play

# Run our agent against 3 rule_based_agents
python main.py play --my-agent taco_kebab_agent

# Train our agent (first N agents in --agents run in training mode)
python main.py play --my-agent taco_kebab_agent --train 1 --n-rounds 1000 --no-gui

# Explicit agent lineup, headless, fast
python main.py play --agents taco_kebab_agent rule_based_agent rule_based_agent rule_based_agent --train 1 --no-gui --n-rounds 1000

# Different scenarios (defined in settings.py): empty, coin-heaven, loot-crate, classic
python main.py play --my-agent taco_kebab_agent --scenario coin-heaven --no-gui

# Reproducible run
python main.py play --my-agent taco_kebab_agent --seed 42

# Replay a saved game
python main.py replay <path-to-replay.pt>

# Run the framework's smoke test (single unittest: play 1 round headless, check logs/game.log written)
python test.py
# or
python -m unittest test.py
```

Dependencies (pygame, tqdm, numpy, scipy, scikit-learn, joblib) are listed unpinned in
`requirements.txt` at the repo root; install with `pip install -r requirements.txt`. They are also
installed via the `Dockerfile` (conda + pip) for the submission image. pygame is optional at
runtime — `fallbacks.py` provides a headless fallback so `--no-gui` runs work without it. There is no
linter or formatter configured in this repo.

## Architecture

### Game loop
`main.py` parses CLI args, builds a `BombeRLeWorld` (`environment.py`) and optionally a `GUI`, then
drives `world_controller()` which calls `world.do_step()` in a loop for `n_rounds`. `settings.py`
holds all game constants (board size 17x17, `MAX_STEPS=400`, bomb power/timer, scenario definitions,
per-step `TIMEOUT=0.5s` for non-training agents).

### Agent interface (what we implement)
Each agent lives in `agent_code/<name>/` and is loaded dynamically by name. Only two files matter:

- **`callbacks.py`** — always loaded.
  - `setup(self)`: one-time init (load model from disk, etc.)
  - `act(self, game_state: dict) -> str`: called every step, must return one of the 6 action
    strings within the timeout.
- **`train.py`** — loaded only when the agent runs with `--train`.
  - `setup_training(self)`: called once after `setup`.
  - `game_events_occurred(self, old_game_state, self_action, new_game_state, events)`: called after
    every step with the events that occurred; this is where reward shaping / replay-buffer updates /
    online learning happen.
  - `end_of_round(self, last_game_state, last_action, events)`: called when the agent dies or the
    round ends; also the natural place to persist the model.

`self` is a per-agent namespace object shared across all these callbacks (plus `self.logger` and
`self.train`) — not a class instance you define, the framework passes it in.

`game_state` (built by `get_state_for_agent` in `environment.py`) is a dict with keys: `round`,
`step`, `field` (np.array of the arena, walls/crates), `self` (own agent state tuple), `others`
(list of other agents' state), `bombs`, `coins`, `explosion_map`, `user_input`.

Framework-tracked events (passed into `game_events_occurred`/`end_of_round`) are the string constants
defined in `events.py` (e.g. `COIN_COLLECTED`, `KILLED_OPPONENT`, `INVALID_ACTION`, `CRATE_DESTROYED`,
`GOT_KILLED`, `SURVIVED_ROUND`, ...).

Reference/example agents to read for patterns: `agent_code/tpl_agent` (course template, the base our
agents are copied from), `agent_code/rule_based_agent` (scripted baseline, used as the default
opponent), `agent_code/coin_collector_agent`, `agent_code/peaceful_agent`, `agent_code/random_agent`.

### `agent_code/taco_kebab_agent` — our agent
`callbacks.py` and `train.py` are implemented against `interface_contract.md` (see below) rather than
`tpl_agent`'s placeholder logic/action order. `callbacks.py` wires `state_to_features` (currently the
`dev_stubs.state_to_features_stub`) into `Model.predict_q`, then picks an action via epsilon-greedy
selection. `train.py` builds n-step TD targets from per-step transitions and feeds them to
`Model.update`, persisting the model to `model_a.npz` at the end of each round.

### Interface contract (`interface_contract.md`) — read before writing agent logic
This is the team's locked cross-member contract (status: locked 12.08.2026) and takes precedence over
whatever the course template (`tpl_agent`) does by default. Key points:
- **Action order is fixed**: `ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']` — note this
  differs from `tpl_agent/callbacks.py`'s own `ACTIONS` order; the contract's order is authoritative
  everywhere an action index / Q-value array is used.
- `state_to_features(game_state: dict) -> np.ndarray` returns a fixed-size `float32` vector (N=34,
  until feature freeze 18.08).
- `Transition` is a fixed namedtuple: `(features, action, reward, next_features, done)`.
- `Model.predict_q(features) -> np.ndarray` shape `(6,)`, `Model.update(batch: list[Transition])`,
  `Model.save/load(path)` — `path` must be relative (absolute paths break the Docker submission test).
  Model A saves as `.npz`, Model B (forest) as `.joblib`.
- Shared dev stubs (e.g. `state_to_features_stub`, `reward_from_events_stub`) belong in
  `agent_code/<our_agent>/dev_stubs.py`, committed so all members import the same stub file during
  parallel development.
- `reward_from_events` must stay cleanly separable from evaluation — official/tournament evaluation
  always runs with rewards off (framework scoring only).

### Team plan (`PLAN.md`)
Three-person team split by **pipeline layer**, not by model, so both models (A: ridge-linear
Q-learning, B: fitted Q-iteration with a regression forest) share the same features/rewards/eval
harness:
- **Member A (Nehir)**: features & state representation (`state_to_features`, symmetry augmentation).
- **Member B (Daniel)**: learning algorithms & models (`Model` class, `train.py` wiring, replay buffer).
- **Member C (Ege)**: rewards, evaluation & ops (`reward_from_events`, `evaluate.py`, experiment log,
  Docker/submission checks).

Constraints that affect any code changes: no multiprocessing in the *final* agent (training-only is
fine); must run on CPU within the 0.5s/step budget; agent logic must live only in
`agent_code/<our_agent>/` (nothing outside it is part of the tournament submission); `settings.py` may
still change until 14.09, so the pipeline needs to keep working if it does.

## Rules from the team plan

1. **All paths must be relative, never absolute.** Absolute paths break the Docker submission test
   (e.g. `Model.save`/`Model.load` paths — see the interface contract above).
2. **No feature may implement logic that directly returns "the best action."** Features describe the
   state (distances, danger, availability, etc.) — the model must learn from features, not the other
   way around.
3. **Any modification to `settings.py` or to scenarios for speeding up training must be made on a
   branch called `training-env`, never on `master`, and must never end up in the final submission
   zip.**
