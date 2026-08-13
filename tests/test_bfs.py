"""Tests for bfs.py, above all for the bomb timing.

An off-by-one in the danger offsets looks exactly like "the agent has not
learned to escape yet", so it is the cheapest thing to test and the most
expensive thing to miss.

Run from the repository root:  python -m pytest tests
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_code.taco_kebab_agent.bfs import (  # noqa: E402
    ACTIONS, BOMB_POWER, BOMB_TIMER, EXPLOSION_TIMER, NEVER, bfs_direction,
    blast_coords, danger_map, find_escape, first_lethal, free_tiles, safe_from,
    tiles_adjacent_to)

WALL, FREE, CRATE = -1, 0, 1


def make_state(picture, bomb_timers=None):
    """Build a game_state from an ASCII picture, so a test reads like its board.

        #  wall     .  free      c  crate    o  coin
        a  us       e  opponent  B  bomb     x  burning tile

    Rows are written top to bottom, so picture[row][col] is field[col, row].
    """
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
        'self': ('taco_kebab_agent', 0, True, self_pos),
        'others': [(f'opp{i}', 0, True, p) for i, p in enumerate(others)],
        'bombs': list(zip(bombs, timers)),
        'coins': coins,
        'explosion_map': explosion_map,
        'user_input': None,
    }


def first_in_action_order(*names):
    """Which of `names` BFS picks when several first steps are equally short."""
    return next(a for a in ACTIONS[:4] if a in names)


# --- blast geometry -------------------------------------------------------

def test_blast_reaches_bomb_power_in_open_space():
    st = make_state("""
#########
#.......#
#.......#
#...a...#
#.......#
#########
""")
    coords = set(blast_coords(st['field'], (4, 3)))
    assert (4, 3) in coords
    for i in range(1, BOMB_POWER + 1):
        assert (4 + i, 3) in coords
        assert (4 - i, 3) in coords
    assert (4, 5) not in coords                  # the wall row stops that arm


def test_blast_is_stopped_by_walls_but_not_by_crates():
    st = make_state("""
#######
#.c.#.#
#..a..#
#.....#
#######
""")
    field = st['field']
    coords = set(blast_coords(field, (3, 2)))
    assert (2, 2) in coords and (1, 2) in coords

    # The crate burns instead of shielding what is behind it.
    assert (2, 1) in set(blast_coords(field, (2, 2)))

    # A wall stops the arm.
    assert (4, 1) not in set(blast_coords(field, (3, 1)))


# --- danger timing --------------------------------------------------------

def test_bomb_with_timer_t_is_lethal_at_offsets_t_and_t_plus_one():
    st = make_state("""
#######
#a....#
#.....#
#######
""")
    st['bombs'] = [((1, 1), 2)]
    lethal = danger_map(st['field'], st['bombs'], st['explosion_map'])

    assert not lethal[0][1, 1]
    assert not lethal[1][1, 1]
    assert lethal[2][1, 1]
    assert lethal[3][1, 1]
    assert not lethal[4][1, 1]


def test_timer_zero_bomb_kills_this_step():
    """Observed timer 0: the move being chosen now is the last one."""
    st = make_state("""
#######
#a....#
#######
""")
    st['bombs'] = [((1, 1), 0)]
    lethal = danger_map(st['field'], st['bombs'], st['explosion_map'])
    assert lethal[0][1, 1]
    assert lethal[0][2, 1]


def test_burning_tile_is_lethal_now_and_safe_next_step():
    """explosion_map >= 1 blocks the immediate move only; 0 is harmless smoke."""
    st = make_state("""
#######
#a.x..#
#######
""")
    lethal = danger_map(st['field'], st['bombs'], st['explosion_map'])
    assert lethal[0][3, 1]
    assert not lethal[1][3, 1]


def test_freshly_dropped_bomb_detonates_at_offset_bomb_timer():
    st = make_state("""
#######
#a....#
#######
""")
    lethal = danger_map(st['field'], [], st['explosion_map'], extra_bomb=(1, 1))
    for k in range(BOMB_TIMER):
        assert not lethal[k][1, 1], f'should still be safe at offset {k}'
    for k in range(BOMB_TIMER, BOMB_TIMER + EXPLOSION_TIMER):
        assert lethal[k][1, 1], f'should be lethal at offset {k}'


def test_first_lethal_and_safe_from_agree():
    st = make_state("""
#######
#a....#
#######
""")
    st['bombs'] = [((1, 1), 2)]
    lethal = danger_map(st['field'], st['bombs'], st['explosion_map'])
    first = first_lethal(lethal)
    safe = safe_from(lethal)

    assert first[1, 1] == 2
    assert first[5, 1] == NEVER
    assert not safe[0][1, 1]
    assert safe[4][1, 1]
    assert safe[0][5, 1]


# --- escape search --------------------------------------------------------

def test_one_step_out_of_the_blast_is_enough():
    st = make_state("""
##########
#B..a....#
##########
""", bomb_timers=[3])
    free = free_tiles(st['field'], st['bombs'])
    lethal = danger_map(st['field'], st['bombs'], st['explosion_map'])
    ok, steps = find_escape(free, (4, 1), lethal, safe_from(lethal))
    assert ok and steps == 1


def test_no_escape_from_a_dead_end():
    """We bomb ourselves at the closed end of a short pocket."""
    st = make_state("""
#####
#a###
#.###
#.###
#####
""")
    here = (1, 1)
    free = free_tiles(st['field'], [], extra_bomb=here)
    lethal = danger_map(st['field'], [], st['explosion_map'], extra_bomb=here)
    ok, steps = find_escape(free, here, lethal, safe_from(lethal))
    assert not ok and steps == -1


def test_escape_when_the_pocket_is_deeper_than_the_blast():
    st = make_state("""
######
#a####
#.####
#.####
#.####
#.####
######
""")
    here = (1, 1)
    free = free_tiles(st['field'], [], extra_bomb=here)
    lethal = danger_map(st['field'], [], st['explosion_map'], extra_bomb=here)
    ok, steps = find_escape(free, here, lethal, safe_from(lethal))
    assert ok and steps == 4


def test_standing_on_a_safe_tile_needs_zero_steps():
    st = make_state("""
##########
#B.......#
#.......a#
##########
""", bomb_timers=[3])
    free = free_tiles(st['field'], st['bombs'])
    lethal = danger_map(st['field'], st['bombs'], st['explosion_map'])
    ok, steps = find_escape(free, (8, 2), lethal, safe_from(lethal))
    assert ok and steps == 0


def test_opponent_blocks_only_the_first_move():
    """An opponent in the way costs a step but does not trap us permanently."""
    st = make_state("""
##########
#B..ae...#
##########
""", bomb_timers=[3])
    free = free_tiles(st['field'], st['bombs'])
    lethal = danger_map(st['field'], st['bombs'], st['explosion_map'])
    ok, steps = find_escape(free, (4, 1), lethal, safe_from(lethal),
                            blocked_now={(5, 1)})
    assert ok and steps == 2


# --- target search --------------------------------------------------------

def test_bfs_finds_nearest_coin_and_reports_first_step():
    st = make_state("""
#######
#a....#
#.....#
#...o.#
#######
""")
    free = free_tiles(st['field'], [])
    d, dist = bfs_direction(free, (1, 1), st['coins'])
    assert dist == 5
    assert ACTIONS[d] == first_in_action_order('RIGHT', 'DOWN')


def test_bfs_routes_around_walls():
    st = make_state("""
#######
#a#o..#
#.#...#
#.....#
#######
""")
    free = free_tiles(st['field'], [])
    d, dist = bfs_direction(free, (1, 1), st['coins'])
    assert ACTIONS[d] == 'DOWN'
    assert dist == 6


def test_bfs_returns_no_target_when_unreachable():
    st = make_state("""
#######
#a#.o.#
#.#...#
###...#
#######
""")
    free = free_tiles(st['field'], [])
    assert bfs_direction(free, (1, 1), st['coins']) == (-1, -1)


def test_crate_targets_are_the_free_tiles_beside_a_crate():
    st = make_state("""
#######
#a....#
#..c..#
#######
""")
    free = free_tiles(st['field'], [])
    targets = tiles_adjacent_to(st['field'], CRATE, free)
    assert (3, 1) in targets and (2, 2) in targets and (4, 2) in targets
    assert (3, 2) not in targets
    d, dist = bfs_direction(free, (1, 1), targets)
    assert dist == 2


def test_bomb_tiles_are_not_walkable():
    st = make_state("""
#######
#aB...#
#######
""", bomb_timers=[3])
    free = free_tiles(st['field'], st['bombs'])
    assert not free[2, 1]
