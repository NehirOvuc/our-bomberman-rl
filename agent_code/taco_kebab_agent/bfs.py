"""Grid reasoning for the agent: passability, danger over time, shortest paths.

Board convention (environment.py): field[x, y] with x = column, y = row,
-1 = wall, 0 = free, 1 = crate.

Nothing in this module returns an action.
"""

from collections import deque

import numpy as np

import settings as s

# Action order, interface_contract.md section 1. This is the only place it is
# written down: DIRECTIONS is derived from it, and everything that indexes an
# action or a Q-value follows from here.
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']

# y increases downwards, so UP is -1 in y.
_STEP = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}

#: Direction vectors, indexed like ACTIONS[0:4].
DIRECTIONS = tuple(_STEP[a] for a in ACTIONS[:4])

#: Indices of the actions that are not moves (BOMB, WAIT).
NON_DIRECTIONAL = tuple(i for i, a in enumerate(ACTIONS) if a not in _STEP)

# Fail at import rather than silently mis-indexing later.
assert all(a in _STEP for a in ACTIONS[:4]), 'ACTIONS[0:4] must be the four moves'
assert len(ACTIONS) == 6, 'the framework has exactly six actions'

BOMB_POWER = s.BOMB_POWER
BOMB_TIMER = s.BOMB_TIMER
EXPLOSION_TIMER = s.EXPLOSION_TIMER

#: A bomb dropped now detonates at offset BOMB_TIMER and burns for
#: EXPLOSION_TIMER steps, so nothing beyond this can matter.
HORIZON = BOMB_TIMER + EXPLOSION_TIMER

#: "This tile never becomes lethal within the horizon."
NEVER = HORIZON + 1


def blast_coords(field, pos, power=BOMB_POWER):
    """Tiles covered by a bomb at pos, mirroring items.Bomb.get_blast_coords.

    Walls stop an arm, crates do not. Walls never change during a game, so a
    footprint computed now stays valid for the bomb's whole lifetime.
    """
    x, y = pos
    coords = [(x, y)]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for i in range(1, power + 1):
            nx, ny = x + i * dx, y + i * dy
            if field[nx, ny] == -1:
                break
            coords.append((nx, ny))
    return coords


def danger_map(field, bombs, explosion_map, extra_bomb=None):
    """lethal[k, x, y]: ending step offset k on (x, y) is fatal. Offset 0 is now.

    Timing follows environment.do_step. A bomb is created with timer BOMB_TIMER
    and decremented in the same step, so an observed timer is 0..3. A bomb
    observed with timer t detonates at offset t and its footprint stays lethal
    for EXPLOSION_TIMER steps, i.e. offsets t .. t + EXPLOSION_TIMER - 1.

    A bomb dropped this step is not in `bombs` yet; pass it as `extra_bomb` and
    it detonates at offset BOMB_TIMER.

    explosion_map >= 1 means the tile is still burning for the current move
    only; a 0 there is harmless smoke.
    """
    lethal = np.zeros((HORIZON + 1,) + field.shape, dtype=bool)

    if explosion_map is not None:
        lethal[0] |= (explosion_map >= 1)

    def add(pos, detonation_offset):
        coords = blast_coords(field, pos)
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        for k in range(detonation_offset, detonation_offset + EXPLOSION_TIMER):
            if 0 <= k <= HORIZON:
                lethal[k, xs, ys] = True

    for pos, timer in bombs:
        add(pos, timer)

    if extra_bomb is not None:
        add(extra_bomb, BOMB_TIMER)

    return lethal


def first_lethal(lethal):
    """Per tile, the smallest offset at which it is lethal (NEVER if none)."""
    any_lethal = lethal.any(axis=0)
    first = np.argmax(lethal, axis=0)
    return np.where(any_lethal, first, NEVER)


def safe_from(lethal):
    """out[k, x, y]: the tile is lethal at no offset j >= k.

    Reaching such a tile at offset k means we can stay there and survive.
    """
    out = np.empty_like(lethal)
    running = np.zeros(lethal.shape[1:], dtype=bool)
    for k in range(lethal.shape[0] - 1, -1, -1):
        running = running | lethal[k]
        out[k] = ~running
    return out


def free_tiles(field, bombs=(), extra_bomb=None):
    """Tiles we may walk onto: empty floor with no bomb on them.

    Opponents are excluded because they move; callers that care about the
    immediate step pass them separately as `blocked_now`.
    """
    free = (field == 0)
    for pos, _ in bombs:
        free[pos] = False
    if extra_bomb is not None:
        free[extra_bomb] = False
    return free


def find_escape(free, start, lethal, safe, blocked_now=(), max_steps=HORIZON):
    """Can we reach a permanently safe tile? Returns (bool, n_steps), -1 on failure.

    Searches over (tile, offset) pairs: at each offset we may wait or step to an
    adjacent free tile, and the tile we end on must not be lethal at that
    offset. Opponents block the first move only, since they move as well and
    freezing them would invent dead ends.
    """
    if safe[0][start]:
        return True, 0

    blocked_now = set(blocked_now)
    seen = {(start, 0)}
    frontier = deque([(start, 0)])

    while frontier:
        (x, y), k = frontier.popleft()
        if k >= max_steps:
            continue
        for nxt in ((x, y), (x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)):
            if nxt != (x, y):
                if not free[nxt]:
                    continue
                if k == 0 and nxt in blocked_now:
                    continue
            if lethal[k][nxt]:
                continue
            if (nxt, k + 1) in seen:
                continue
            if safe[k + 1][nxt]:
                return True, k + 1
            seen.add((nxt, k + 1))
            frontier.append((nxt, k + 1))

    return False, -1


def bfs_direction(free, start, targets):
    """Nearest target by BFS. Returns (direction index into DIRECTIONS, distance).

    (-1, -1) means no target is reachable. Target tiles may be entered even if
    they are not free (an opponent's tile), but are never expanded through.

    Deterministic, unlike rule_based_agent's look_for_targets: neighbours are
    always visited in ACTIONS order, so ties break the same way every time.
    """
    targets = set(targets)
    if not targets:
        return -1, -1
    if start in targets:
        return -1, 0

    seen = {start}
    frontier = deque()
    for d, (dx, dy) in enumerate(DIRECTIONS):
        nxt = (start[0] + dx, start[1] + dy)
        if nxt in seen:
            continue
        if not (free[nxt] or nxt in targets):
            continue
        if nxt in targets:
            return d, 1
        seen.add(nxt)
        frontier.append((nxt, d, 1))

    while frontier:
        (x, y), d0, dist = frontier.popleft()
        for dx, dy in DIRECTIONS:
            nxt = (x + dx, y + dy)
            if nxt in seen:
                continue
            if not (free[nxt] or nxt in targets):
                continue
            if nxt in targets:
                return d0, dist + 1
            seen.add(nxt)
            frontier.append((nxt, d0, dist + 1))

    return -1, -1


def tiles_adjacent_to(field, value, free):
    """Free tiles orthogonally adjacent to a tile holding `value`.

    A crate cannot be stood on, so "go to the nearest crate" means "go to a
    free tile touching a crate".
    """
    mask = (field == value)
    out = set()
    xs, ys = np.nonzero(mask)
    for x, y in zip(xs.tolist(), ys.tolist()):
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1] and free[nx, ny]:
                out.add((nx, ny))
    return out
