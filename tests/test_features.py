"""Tests for features.py.

Two kinds of test here. The contract ones (shape, dtype, bounds, None, and
determinism) protect everyone downstream: a feature vector that quietly
changes length or dtype breaks the model fit rather than raising. The rest
check that individual components mean what their name says.

Run from the repository root:  python -m pytest tests
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helpers import make_state  # noqa: E402

from agent_code.taco_kebab_agent.features import (  # noqa: E402
    DTYPE, FEATURE_DIM, FEATURE_NAMES, MAX_DIST, state_to_features)

OPEN_BOARD = """
#######
#a....#
#.....#
#.....#
#######
"""


def f(phi, name):
    """Look a feature up by name instead of by magic index."""
    return phi[FEATURE_NAMES.index(name)]


# --- contract -------------------------------------------------------------

def test_shape_and_dtype():
    phi = state_to_features(make_state(OPEN_BOARD))
    assert phi.shape == (FEATURE_DIM,)
    assert phi.dtype == DTYPE


def test_names_match_the_dimension_and_are_unique():
    assert len(FEATURE_NAMES) == FEATURE_DIM
    assert len(set(FEATURE_NAMES)) == FEATURE_DIM


def test_none_state_gives_none():
    """train.py uses this to mark terminal transitions."""
    assert state_to_features(None) is None


def test_every_component_is_bounded_in_zero_one():
    boards = [OPEN_BOARD, """
#########
#a.c.o.e#
#.cc..B.#
#..o.c..#
#########
"""]
    for board in boards:
        phi = state_to_features(make_state(board))
        assert np.all(phi >= 0.0) and np.all(phi <= 1.0)


def test_extraction_is_deterministic():
    st = make_state("""
#########
#a.c.o.e#
#.cc..B.#
#########
""")
    assert np.array_equal(state_to_features(st), state_to_features(st))


def test_bias_is_one():
    assert f(state_to_features(make_state(OPEN_BOARD)), 'bias') == 1.0


# --- what is next to me ---------------------------------------------------

def test_free_flags_follow_the_walls():
    phi = state_to_features(make_state(OPEN_BOARD))   # agent in the top-left
    assert f(phi, 'free_UP') == 0.0
    assert f(phi, 'free_LEFT') == 0.0
    assert f(phi, 'free_DOWN') == 1.0
    assert f(phi, 'free_RIGHT') == 1.0


def test_an_opponent_makes_a_tile_not_free():
    phi = state_to_features(make_state("""
#######
#ae...#
#.....#
#######
"""))
    assert f(phi, 'free_RIGHT') == 0.0


def test_adjacent_crate_flag():
    phi = state_to_features(make_state("""
#######
#ac...#
#.....#
#######
"""))
    assert f(phi, 'adj_crate_RIGHT') == 1.0
    assert f(phi, 'adj_crate_DOWN') == 0.0


# --- danger ---------------------------------------------------------------

def test_no_bomb_means_no_danger_anywhere():
    phi = state_to_features(make_state(OPEN_BOARD))
    assert f(phi, 'danger_HERE') == 0.0
    assert f(phi, 'in_danger') == 0.0
    assert f(phi, 'escape_possible') == 1.0


def test_standing_in_a_blast_sets_the_danger_clock():
    """Timer 2 means lethal in two steps, so urgency is 1 / (1 + 2)."""
    st = make_state("""
#########
#aB.....#
#.......#
#########
""", bomb_timers=[2])
    phi = state_to_features(st)
    assert f(phi, 'in_danger') == 1.0
    assert f(phi, 'danger_HERE') == np.float32(1.0 / 3.0)


def test_a_burning_tile_is_maximally_urgent():
    phi = state_to_features(make_state("""
#######
#ax...#
#.....#
#######
"""))
    assert f(phi, 'danger_RIGHT') == 1.0


def test_no_escape_when_the_only_free_tile_is_in_the_blast():
    """Bomb next to us, nowhere to step: the blast covers the whole pocket."""
    phi = state_to_features(make_state("""
#####
#aB##
#####
""", bomb_timers=[1]))
    assert f(phi, 'in_danger') == 1.0
    assert f(phi, 'danger_HERE') == np.float32(0.5)
    assert f(phi, 'escape_possible') == 0.0


# --- targets --------------------------------------------------------------

def test_coin_direction_is_one_hot_and_distance_decreases_with_range():
    near = state_to_features(make_state("""
#######
#a.o..#
#.....#
#######
"""))
    far = state_to_features(make_state("""
#########
#a.....o#
#.......#
#########
"""))
    coin_dirs = [f(near, n) for n in FEATURE_NAMES if n.startswith('to_coin_')]
    assert sum(coin_dirs) == 1.0
    assert f(near, 'to_coin_RIGHT') == 1.0
    assert f(near, 'coin_dist') == np.float32(1.0 - 2.0 / MAX_DIST)
    assert f(far, 'coin_dist') < f(near, 'coin_dist')


def test_no_coin_means_no_direction_and_zero_distance():
    phi = state_to_features(make_state(OPEN_BOARD))
    assert sum(f(phi, n) for n in FEATURE_NAMES if n.startswith('to_coin_')) == 0.0
    assert f(phi, 'coin_dist') == 0.0


def test_crate_direction_points_at_the_nearest_crate():
    phi = state_to_features(make_state("""
#######
#a....#
#.....#
#..c..#
#######
"""))
    assert sum(f(phi, n) for n in FEATURE_NAMES if n.startswith('to_crate_')) == 1.0


def test_opponent_direction_and_distance():
    phi = state_to_features(make_state("""
#######
#a...e#
#.....#
#######
"""))
    assert f(phi, 'to_opp_RIGHT') == 1.0
    assert f(phi, 'opp_dist') == np.float32(1.0 - 4.0 / MAX_DIST)


# --- what a bomb here would do --------------------------------------------

def test_bomb_features_are_zero_when_no_bomb_is_available():
    phi = state_to_features(make_state(OPEN_BOARD, bombs_left=False))
    assert f(phi, 'bomb_available') == 0.0
    assert f(phi, 'bomb_crate_count') == 0.0
    assert f(phi, 'bomb_escape_possible') == 0.0


def test_bomb_crate_count_scales_with_the_crates_hit():
    one = state_to_features(make_state("""
#######
#ac...#
#.....#
#######
"""))
    two = state_to_features(make_state("""
#######
#ac...#
#c....#
#######
"""))
    assert f(two, 'bomb_crate_count') > f(one, 'bomb_crate_count')
    assert f(one, 'bomb_crate_count') == np.float32(0.25)


def test_bomb_hits_opponent():
    phi = state_to_features(make_state("""
#######
#a.e..#
#.....#
#######
"""))
    assert f(phi, 'bomb_hits_opponent') == 1.0


def test_bombing_a_dead_end_is_not_survivable():
    """The pocket is shorter than the blast, so there is nowhere to run."""
    phi = state_to_features(make_state("""
#####
#a###
#.###
#####
"""))
    assert f(phi, 'bomb_escape_possible') == 0.0


def test_bombing_an_open_board_is_survivable():
    phi = state_to_features(make_state("""
#########
#a......#
#.......#
#.......#
#.......#
#########
"""))
    assert f(phi, 'bomb_escape_possible') == 1.0
