"""Hand-built game states for the tests.

"Bomb two tiles away, one escape route" hard to test.
`make_state` builds one from an ASCII picture instead, so a test reads like
the board it is testing.

    #  wall     .  free      c  crate    o  coin
    a  us       e  opponent  B  bomb     x  burning tile

Rows are written top to bottom, so picture[row][col] is field[col, row].
"""

import numpy as np

WALL, FREE, CRATE = -1, 0, 1


def make_state(picture, bomb_timers=None, bombs_left=True):
    """Build a game_state dict from an ASCII picture."""
    rows = picture.strip('\n').split('\n')
    height, width = len(rows), len(rows[0])
    assert all(len(r) == width for r in rows), 'picture rows must be equal length'

    field = np.zeros((width, height), dtype=int)
    explosion_map = np.zeros((width, height), dtype=float)
    coins, bombs, others = [], [], []
    self_pos = None

    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == '#':
                field[x, y] = WALL
            elif ch == 'c':
                field[x, y] = CRATE
            elif ch == 'o':
                coins.append((x, y))
            elif ch == 'a':
                self_pos = (x, y)
            elif ch == 'e':
                others.append((x, y))
            elif ch == 'B':
                bombs.append((x, y))
            elif ch == 'x':
                explosion_map[x, y] = 1.0
            elif ch != '.':
                raise ValueError(f'unknown character {ch!r}')

    timers = bomb_timers if bomb_timers is not None else [3] * len(bombs)
    assert len(timers) == len(bombs), 'one timer per bomb'
    assert self_pos is not None, "picture must contain 'a'"

    return {
        'round': 1,
        'step': 1,
        'field': field,
        'self': ('taco_kebab_agent', 0, bombs_left, self_pos),
        'others': [(f'opp{i}', 0, True, p) for i, p in enumerate(others)],
        'bombs': list(zip(bombs, timers)),
        'coins': coins,
        'explosion_map': explosion_map,
        'user_input': None,
    }
