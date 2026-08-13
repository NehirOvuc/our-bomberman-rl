"""Feature extraction: game_state -> phi(s), a fixed-length float vector.

This is the shared input to both models, and the state-dependent reward
shaping reads the same quantities from bfs.py, so features and rewards can
never disagree about where the danger is.

Three rules the vector follows (interface_contract.md sections 2 and 3):

1. No feature encodes "the best action". We give a direction to the nearest
   coin and a per-tile danger clock, but never a per-direction "this move
   keeps you alive" flag -- that would solve bomb escape here and leave the
   model nothing to learn.
2. Direction-indexed features come in groups of four in ACTIONS order, so
   symmetry augmentation can permute indices instead of re-extracting.
3. Every component is bounded in [0, 1], so a single ridge lambda suits all
   of them.
"""

import numpy as np

from .bfs import (ACTIONS, DIRECTIONS, NEVER, bfs_direction, blast_coords,
                  danger_map, find_escape, first_lethal, free_tiles, safe_from,
                  tiles_adjacent_to)

#: interface_contract.md section 3.
DTYPE = np.float32

#: Longest distance we bother to distinguish; beyond this, far is far.
MAX_DIST = 20.0

#: Ceiling for the crate counter, chosen to spread it over [0, 1].
MAX_CRATES_PER_BOMB = 4.0

_DIRS = tuple(ACTIONS[:4])


def _named(prefix):
    return [f'{prefix}{d}' for d in _DIRS]


FEATURE_NAMES = [
    *_named('free_'),               # 0-3
    *_named('adj_crate_'),          # 4-7
    *_named('danger_'),             # 8-11
    'danger_HERE',                  # 12
    'in_danger',                    # 13
    'escape_possible',              # 14
    *_named('to_coin_'),            # 15-18
    'coin_dist',                    # 19
    *_named('to_crate_'),           # 20-23
    *_named('to_opp_'),             # 24-27
    'opp_dist',                     # 28
    'bomb_available',               # 29
    'bomb_crate_count',             # 30
    'bomb_hits_opponent',           # 31
    'bomb_escape_possible',         # 32
    'bias',                         # 33
]

#: interface_contract.md section 2. Import this; do not hard-code the number.
FEATURE_DIM = len(FEATURE_NAMES)

#: Start index of each group of four direction-indexed features. symmetry.py
#: permutes exactly these and leaves the rest alone.
FEATURE_GROUPS = {
    'free': 0,
    'adjacent_crate': 4,
    'danger': 8,
    'coin_dir': 15,
    'crate_dir': 20,
    'opp_dir': 24,
}
DIRECTIONAL_OFFSETS = tuple(sorted(FEATURE_GROUPS.values()))


def _urgency(offset):
    """Map "lethal in k steps" to [0, 1]: now -> 1.0, next step -> 0.5, never -> 0.

    The reciprocal shape is deliberate. The gap between dying now and dying
    next step matters enormously; the gap between four and five steps does not.
    """
    if offset >= NEVER:
        return 0.0
    return 1.0 / (1.0 + float(offset))


def state_to_features(game_state: dict) -> np.ndarray:
    """Convert a game state into phi(s), shape (FEATURE_DIM,), dtype float32.

    Returns None when `game_state` is None. The framework passes None before
    the first step and after the agent dies, and train.py uses exactly that to
    mark terminal transitions.
    """
    if game_state is None:
        return None

    field = game_state['field']
    _, _, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    coins = game_state['coins']
    explosion_map = game_state['explosion_map']
    others = [pos for (_, _, _, pos) in game_state['others']]

    here = (x, y)
    phi = np.zeros(FEATURE_DIM, dtype=DTYPE)

    free = free_tiles(field, bombs)
    lethal = danger_map(field, bombs, explosion_map)
    first = first_lethal(lethal)
    safe = safe_from(lethal)
    others_set = set(others)

    # What is next to me, and how urgent is the danger there.
    for d, (dx, dy) in enumerate(DIRECTIONS):
        nxt = (x + dx, y + dy)
        phi[FEATURE_GROUPS['free'] + d] = float(free[nxt] and nxt not in others_set)
        phi[FEATURE_GROUPS['adjacent_crate'] + d] = float(field[nxt] == 1)
        phi[FEATURE_GROUPS['danger'] + d] = _urgency(first[nxt])

    phi[12] = _urgency(first[here])
    phi[13] = float(first[here] < NEVER)

    escape_ok, _ = find_escape(free, here, lethal, safe, blocked_now=others_set)
    phi[14] = float(escape_ok)

    # Where the interesting things are.
    coin_dir, coin_dist = bfs_direction(free, here, coins)
    if coin_dir >= 0:
        phi[FEATURE_GROUPS['coin_dir'] + coin_dir] = 1.0
    phi[19] = 1.0 - min(coin_dist, MAX_DIST) / MAX_DIST if coin_dist >= 0 else 0.0

    crate_targets = tiles_adjacent_to(field, 1, free)
    crate_dir, _ = bfs_direction(free, here, crate_targets)
    if crate_dir >= 0:
        phi[FEATURE_GROUPS['crate_dir'] + crate_dir] = 1.0

    opp_dir, opp_dist = bfs_direction(free, here, others)
    if opp_dir >= 0:
        phi[FEATURE_GROUPS['opp_dir'] + opp_dir] = 1.0
    phi[28] = 1.0 - min(opp_dist, MAX_DIST) / MAX_DIST if opp_dist >= 0 else 0.0

    # What would happen if I dropped a bomb right now.
    phi[29] = float(bombs_left)
    if bombs_left:
        coords = blast_coords(field, here)
        n_crates = sum(1 for c in coords if field[c] == 1)
        phi[30] = min(n_crates, MAX_CRATES_PER_BOMB) / MAX_CRATES_PER_BOMB
        phi[31] = float(any(o in coords for o in others))

        # A hypothetical, not a recommendation: it says bombing here would be
        # survivable, not that it is a good idea. The model still has to learn
        # when to use it.
        v_free = free_tiles(field, bombs, extra_bomb=here)
        v_lethal = danger_map(field, bombs, explosion_map, extra_bomb=here)
        ok, _ = find_escape(v_free, here, v_lethal, safe_from(v_lethal),
                            blocked_now=others_set)
        phi[32] = float(ok)

    phi[33] = 1.0

    return phi


def describe(phi) -> str:
    """Human-readable dump of a feature vector, for debugging."""
    if phi is None:
        return '<terminal state>'
    width = max(len(n) for n in FEATURE_NAMES)
    return '\n'.join(
        f'{name:<{width}} {value: .3f}'
        for name, value in zip(FEATURE_NAMES, np.asarray(phi).ravel()))
