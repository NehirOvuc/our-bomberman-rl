"""Tests for rewards.py. Boards are drawn with helpers.make_state.

Legend: # wall  . free  c crate  o coin  a us  e opponent  B bomb  x burning
"""

import events as e
from agent_code.taco_kebab_agent.rewards import (
    reward_from_events, derive_events, REWARDS,
    MOVED_TOWARDS_COIN, MOVED_AWAY_FROM_COIN, ESCAPED_BLAST, STAYED_IN_BLAST,
)
from helpers import make_state


# --- the plain table -------------------------------------------------------

def test_each_table_event_pays_its_value():
    for event, value in REWARDS.items():
        assert reward_from_events([event], None, None) == value


def test_empty_event_list_gives_zero():
    assert reward_from_events([], None, None) == 0.0


def test_unknown_event_counts_as_zero_not_error():
    assert reward_from_events(['SOMETHING_NEW'], None, None) == 0.0


def test_return_type_is_float():
    # The template's version returned int; the contract says float.
    assert isinstance(reward_from_events([e.WAITED], None, None), float)


# --- the suicide decision --------------------------------------------------

def test_suicide_costs_five_not_ten():
    # A self-kill fires both death events; we count the death once.
    both = [e.KILLED_SELF, e.GOT_KILLED]
    assert reward_from_events(both, None, None) == -5.0


def test_killed_by_opponent_still_costs_five():
    assert reward_from_events([e.GOT_KILLED], None, None) == -5.0


# --- coin shaping ----------------------------------------------------------

def test_step_towards_coin():
    old = make_state("""
#######
#a..o.#
#######
""")
    new = make_state("""
#######
#.a.o.#
#######
""")
    assert derive_events([], old, new) == [MOVED_TOWARDS_COIN]


def test_step_away_from_coin():
    old = make_state("""
#######
#.a.o.#
#######
""")
    new = make_state("""
#######
#a..o.#
#######
""")
    assert derive_events([], old, new) == [MOVED_AWAY_FROM_COIN]


def test_no_coin_shaping_on_the_collection_step():
    # After collecting, the nearest coin is suddenly a farther one. That must
    # not read as "moved away" on the agent's best step.
    old = make_state("""
#######
#a.o.o#
#######
""")
    new = make_state("""
#######
#..a.o#
#######
""")
    r = reward_from_events([e.COIN_COLLECTED], old, new)
    assert r == REWARDS[e.COIN_COLLECTED]


def test_no_coin_shaping_when_no_coin_reachable():
    old = make_state("""
#######
#a....#
#######
""")
    new = make_state("""
#######
#.a...#
#######
""")
    assert derive_events([], old, new) == []


# --- danger shaping --------------------------------------------------------

def test_escaping_the_blast_line():
    # Bomb power is 3 tiles: distance 3 is in range, distance 4 is out.
    old = make_state("""
########
#B..a..#
########
""", bomb_timers=[2])
    new = make_state("""
########
#B...a.#
########
""", bomb_timers=[1])
    assert derive_events([], old, new) == [ESCAPED_BLAST]


def test_staying_in_the_blast_line():
    old = make_state("""
########
#B..a..#
########
""", bomb_timers=[2])
    new = make_state("""
########
#B.a...#
########
""", bomb_timers=[1])
    assert derive_events([], old, new) == [STAYED_IN_BLAST]


def test_walking_into_danger_costs_nothing_yet():
    # Known, accepted property: the contract's two danger events both need the
    # agent to already be in danger. The penalty lands one step later.
    old = make_state("""
########
#B...a.#
########
""", bomb_timers=[2])
    new = make_state("""
########
#B..a..#
########
""", bomb_timers=[1])
    assert derive_events([], old, new) == []


# --- terminal steps --------------------------------------------------------

def test_none_states_derive_nothing():
    assert derive_events([], None, None) == []
    assert derive_events([], make_state("""
###
#a#
###
"""), None) == []


# --- the contract's bound --------------------------------------------------

def test_ordinary_steps_stay_inside_the_bound():
    # Every single-event reward, and the suicide pair, stays in [-5, +5].
    for event in REWARDS:
        assert -5.0 <= reward_from_events([event], None, None) <= 5.0
    assert -5.0 <= reward_from_events([e.KILLED_SELF, e.GOT_KILLED], None, None)
