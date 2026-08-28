"""Turn the framework's events into a training reward. Contract §5.

The framework tells us what happened each step as a list of event strings
(events.py). This module turns that list into one float that training uses.
Evaluation never uses it: evaluate.py runs on the framework's real scoring
only, so nothing here may be needed to just play a game.

Four events from the contract's reward table do not exist in the framework:
MOVED_TOWARDS_COIN, MOVED_AWAY_FROM_COIN, ESCAPED_BLAST, STAYED_IN_BLAST.
We work those out ourselves by comparing the old and new game state. That is
why reward_from_events takes both states. All distance and danger questions
are answered by bfs.py, the same module features.py uses, so the rewards and
the features can never disagree about where a coin or a danger is.

Decisions made here (26-28.08.2026, details in docs/open-decisions.md):
- A suicide fires both KILLED_SELF and GOT_KILLED. We count only KILLED_SELF,
  so dying costs -5 once, no matter who caused it.
- Danger means: my own tile becomes lethal at some upcoming step, judged by
  bfs.danger_map. Same definition as the in_danger feature.
- The event values are exactly the contract's starting table. Tuning comes
  later, from training curves, and gets logged in the experiment log.

The total is roughly bounded in [-5, +5] per step. "Roughly" because a death
plus a small shaping penalty can give e.g. -5.3; the contract says roughly
and we prefer that over clamping, which would silently distort other sums.
"""

import events as e

from .bfs import bfs_direction, danger_map, first_lethal, free_tiles, NEVER

# The four events we derive ourselves. Plain strings, same style as events.py.
MOVED_TOWARDS_COIN = 'MOVED_TOWARDS_COIN'
MOVED_AWAY_FROM_COIN = 'MOVED_AWAY_FROM_COIN'
ESCAPED_BLAST = 'ESCAPED_BLAST'
STAYED_IN_BLAST = 'STAYED_IN_BLAST'

# The contract's starting table (§5). Events not listed here count as 0.
REWARDS = {
    e.COIN_COLLECTED: 1.0,
    e.KILLED_OPPONENT: 5.0,
    e.KILLED_SELF: -5.0,
    e.GOT_KILLED: -5.0,
    e.CRATE_DESTROYED: 0.2,
    e.INVALID_ACTION: -0.1,
    e.WAITED: -0.05,
    MOVED_TOWARDS_COIN: 0.1,
    MOVED_AWAY_FROM_COIN: -0.1,
    ESCAPED_BLAST: 0.3,
    STAYED_IN_BLAST: -0.3,
}


def _coin_distance(game_state):
    """Steps to the nearest reachable coin, or -1 if there is none."""
    field = game_state['field']
    _, _, _, here = game_state['self']
    free = free_tiles(field, game_state['bombs'])
    _, dist = bfs_direction(free, here, game_state['coins'])
    return dist


def _in_danger(game_state):
    """True if my tile is going to be lethal at some upcoming step."""
    field = game_state['field']
    _, _, _, here = game_state['self']
    lethal = danger_map(field, game_state['bombs'], game_state['explosion_map'])
    return first_lethal(lethal)[here] < NEVER


def derive_events(events, old_game_state, new_game_state):
    """The four contract events the framework does not provide.

    Returns a list with zero or more of MOVED_TOWARDS_COIN,
    MOVED_AWAY_FROM_COIN, ESCAPED_BLAST, STAYED_IN_BLAST. Empty when either
    state is None (first step of a round, or the agent just died).
    """
    if old_game_state is None or new_game_state is None:
        return []

    derived = []

    # Coin shaping. Skipped on the step a coin was collected: the nearest
    # coin then suddenly becomes a farther one, which would look like moving
    # away right when the agent did the best possible thing.
    if e.COIN_COLLECTED not in events:
        old_dist = _coin_distance(old_game_state)
        new_dist = _coin_distance(new_game_state)
        # Only compare when a coin was reachable in both states. If one side
        # is -1 the two numbers mean different things and comparing is unfair.
        if old_dist >= 0 and new_dist >= 0:
            if new_dist < old_dist:
                derived.append(MOVED_TOWARDS_COIN)
            elif new_dist > old_dist:
                derived.append(MOVED_AWAY_FROM_COIN)

    # Danger shaping: was my tile due to explode, and is it still?
    was = _in_danger(old_game_state)
    now = _in_danger(new_game_state)
    if was and not now:
        derived.append(ESCAPED_BLAST)
    elif was and now:
        derived.append(STAYED_IN_BLAST)

    return derived


def reward_from_events(events, old_game_state, new_game_state) -> float:
    """Total reward for one step. Signature fixed by contract §5.

    Unknown events contribute 0, so a framework event we chose not to
    reward never crashes training.
    """
    events = list(events)

    # A suicide fires both death events. Count the death once: keep
    # KILLED_SELF, drop GOT_KILLED.
    if e.KILLED_SELF in events:
        events = [ev for ev in events if ev != e.GOT_KILLED]

    events += derive_events(events, old_game_state, new_game_state)

    return float(sum(REWARDS.get(ev, 0.0) for ev in events))
