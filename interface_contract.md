# Interface Contract — Bomberman RL

**Status: LOCKED — 12.08.2026.** Changes after this point must be flagged in standup before merging.

**What is a stub?** A stub is a fake, temporary version of a function — same name, same inputs, same outputs, but with no real logic inside (e.g. it returns random numbers instead of actually computing anything). It exists so that whoever depends on that function can start building and testing their own code *today*, without waiting for the real version to be finished. Once the real function is ready, you just swap the import — nothing else in your code has to change, because the "shape" (signature) stayed identical.

**How to use this file:** Keep a copy in your working repo. When generating code with Claude Code, point it at this file explicitly — e.g. *"Follow the signatures and values in interface_contract.md, don't invent your own."*

| Role | Owner | Scope |
|---|---|---|
| Member A | **Nehir** | Features & state representation |
| Member B | **Daniel** | Learning algorithms & models |
| Member C | **Ege** | Rewards, evaluation & operations |

---

## 1. 6-Action Space Order

**Impacts:** every function that returns or reads an action or a Q-value array — `state_to_features` (indirectly, via action-dependent features), `Model.predict_q`, `act()`, `reward_from_events`.

**Value chosen:**
```python
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
```
Fixed everywhere in the codebase — index `i` always means `ACTIONS[i]`.

---

## 2. Nehir → Feature Vector Size

**Impacts:** `state_to_features(game_state: dict) -> np.ndarray`

**Value chosen for N:** `34`

**Stub for Daniel and Ege to build against until feature freeze (18.08):**
```python
def state_to_features_stub(game_state: dict) -> np.ndarray:
    return np.random.randn(34).astype(np.float32)
```

**Constraint:** must run well within the 0.5s/step budget — Nehir benchmarks this in isolation.

---

## 3. Nehir → Data Type Selection

**Impacts:** `state_to_features(game_state: dict) -> np.ndarray` (return dtype)

**Value chosen:** `np.float32` — compatible with numpy/sklearn, half the memory of `float64`, no meaningful precision loss for this use case.

---

## 4. Daniel → Transition's Fixed Structure

**Impacts:** `Model.update(self, batch: list[Transition]) -> None`

**Value chosen:**
```python
Transition = namedtuple('Transition', [
    'features',       # np.ndarray (34,)
    'action',         # str, one of ACTIONS
    'reward',         # float — output of reward_from_events
    'next_features',  # np.ndarray (34,) or None if terminal
    'done',           # bool
])
```

---

## 5. Ege → Reward Scale/Range Shaping

**Impacts:** `reward_from_events(events, old_game_state, new_game_state) -> float`

**Value chosen:** roughly bounded in `[-5, +5]` per step. Starting table (Ege owns final tuning based on training curves):

| Event | Reward |
|---|---|
| `COIN_COLLECTED` | `+1.0` |
| `KILLED_OPPONENT` | `+5.0` |
| `KILLED_SELF` | `-5.0` |
| `GOT_KILLED` | `-5.0` |
| `CRATE_DESTROYED` | `+0.2` |
| `INVALID_ACTION` | `-0.1` |
| `WAITED` | `-0.05` |
| `MOVED_TOWARDS_COIN` / `MOVED_AWAY_FROM_COIN` | `+0.1` / `-0.1` |
| `ESCAPED_BLAST` / `STAYED_IN_BLAST` | `+0.3` / `-0.3` |

**Stub for Daniel to build against until Ege's table is wired in:**
```python
def reward_from_events_stub(events, old_game_state=None, new_game_state=None) -> float:
    return float(len(events))  # placeholder, not meaningful
```

**Constraint:** evaluation always runs with rewards **off** (framework's real scoring only) — this function must stay cleanly separable from `evaluate.py`.

---

## 6. Daniel → Model Save Format

**Impacts:** `Model.save(self, path: str) -> None` / `Model.load(self, path: str) -> None`

**Value chosen:**
- Model A (ridge-linear): `.npz`
- Model B (forest): `joblib` (`.joblib`)

`joblib` is needed at inference time too (to `load()` the forest in the tournament), so it must be listed in `requirements.txt`. This is a serialization dependency, not the multiprocessing use that's banned for the final agent — worth a one-line clarification in the report so the two aren't confused.

`path` must be relative — absolute paths break the Docker submission test.

---

## 7. Stub Location

**Value chosen:** all stubs live in one shared file, `agent_code/<our_agent>/dev_stubs.py`, committed to the team's GitHub repository — not a personal or local-only file. Once someone creates it and pushes it, everyone else pulls it into their own local copy, so all three branches import from the exact same file during parallel development. This makes it one place to remove stub imports once the real functions are ready.

---

## Reference: other function signatures

Not open decisions, but part of the contract — Daniel and Nehir need these exact shapes to build against.

**`Model.predict_q(self, features: np.ndarray) -> np.ndarray`** (Daniel)
Returns estimated Q-value for each of the 6 actions. Shape `(6,)`, index `i` = `ACTIONS[i]`.

**`apply_symmetry(features: np.ndarray, action: str | None, transform_id: int) -> tuple[np.ndarray, str | None]`** (Nehir)
Applies one of the 8 dihedral transforms to a feature vector, and the matching permuted action if one is given. `transform_id` is an integer in `[0, 7]`. Daniel decides whether/when to call this inside `update()`.

**Framework callbacks** (fixed by the course, not by us):
```python
# callbacks.py — always loaded
def setup(self) -> None: ...
def act(self, game_state: dict) -> str: ...

# train.py — loaded only when self.train == True
def setup_training(self) -> None: ...
def game_events_occurred(self, old_game_state, self_action, new_game_state, events) -> None: ...
def end_of_round(self, last_game_state, last_action, events) -> None: ...
```
`act()` calls `state_to_features` → `Model.predict_q` → picks an action. `game_events_occurred` calls `reward_from_events` → builds a `Transition` → periodically calls `Model.update`.
