"""Tests for symmetry.py.

The one that matters is test_features_are_equivariant: it transforms the board
geometrically, extracts features from scratch, and checks the result equals
the cheap index permutation. Everything else here is bookkeeping; that test is
the only reason the fast path can be trusted, and augmenting training data
with a wrong permutation corrupts every sample without ever raising.

Run from the repository root:  python -m pytest tests
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helpers import make_state  # noqa: E402

from agent_code.taco_kebab_agent.bfs import ACTIONS  # noqa: E402
from agent_code.taco_kebab_agent.features import (  # noqa: E402
    FEATURE_DIM, FEATURE_NAMES, state_to_features)
from agent_code.taco_kebab_agent.model import Model, Transition  # noqa: E402
from agent_code.taco_kebab_agent.symmetry import (  # noqa: E402
    ACTION_PERM, DIR_PERM, IDENTITY, N_TRANSFORMS, apply_symmetry,
    augment_batch, augment_transitions, inverse_transform_action, map_point,
    transform_action, transform_features, transform_state)


#: Boards where the nearest coin, crate and opponent each have a *unique*
#: shortest first step: they sit on the agent's own row or column. See
#: test_only_the_target_directions_can_break_ties for why that matters.
BOARDS = [
    """
#########
#.......#
#.......#
#.......#
#e..a..o#
#.......#
#...c...#
#.......#
#########
""",
    """
#########
#.......#
#...B...#
#.......#
#e..a..o#
#.......#
#...c...#
#.......#
#########
""",
    """
#########
#a......#
#####.###
#.....o.#
#.###.###
#.c.....#
#.......#
#....e..#
#########
""",
]

#: Same idea, but every target sits diagonally, so two first steps are equally
#: short and the deterministic tie-break has to pick one.
TIED_BOARD = """
#########
#a......#
#.c.....#
#....o..#
#.......#
#.....e.#
#.......#
#.......#
#########
"""

TARGET_DIRECTION_FEATURES = [i for i, n in enumerate(FEATURE_NAMES)
                             if n.startswith(('to_coin_', 'to_crate_', 'to_opp_'))]


# --- the property that justifies the fast path ----------------------------

def test_features_are_equivariant():
    """phi(T s) == T phi(s) for all eight T, on real boards.

    The left side re-extracts from a geometrically rotated board; the right
    side only permutes indices. They must agree, or augmentation is lying.
    """
    for picture in BOARDS:
        st = make_state(picture)
        phi = state_to_features(st)
        for t in range(N_TRANSFORMS):
            from_board = state_to_features(transform_state(st, t))
            from_permutation = transform_features(phi, t)
            assert np.array_equal(from_board, from_permutation), \
                f'transform {t} disagrees'


def test_only_the_target_directions_can_break_ties():
    """Everything except the to_* one-hots is equivariant on *any* board.

    When two shortest paths are equally long, bfs_direction breaks the tie by
    ACTIONS order, and that order is not itself rotation-invariant: rotating
    the board can hand the tie to the other direction. So on a tied board the
    permuted vector and the re-extracted one may disagree -- but only in the
    to_coin_/to_crate_/to_opp_ groups, and both answers are still shortest
    first steps. This test pins that the damage is confined there; the boards
    used above avoid ties so the full property can be checked exactly.
    """
    st = make_state(TIED_BOARD)
    phi = state_to_features(st)
    rest = [i for i in range(FEATURE_DIM) if i not in TARGET_DIRECTION_FEATURES]
    for t in range(N_TRANSFORMS):
        from_board = state_to_features(transform_state(st, t))
        from_permutation = transform_features(phi, t)
        assert np.array_equal(from_board[rest], from_permutation[rest]), \
            f'transform {t} differs outside the target-direction groups'


# --- the transforms form the group we think they do -----------------------

def test_identity_changes_nothing():
    phi = state_to_features(make_state(BOARDS[0]))
    assert np.array_equal(transform_features(phi, IDENTITY), phi)
    for a in ACTIONS:
        assert transform_action(a, IDENTITY) == a


def test_there_are_eight_distinct_point_maps():
    """D4 has eight elements; if two transforms coincide we built one wrong."""
    n = 9
    images = {tuple(map_point(x, y, n, t) for x in range(n) for y in range(n))
              for t in range(N_TRANSFORMS)}
    assert len(images) == N_TRANSFORMS


def test_direction_permutations_are_bijections():
    for t in range(N_TRANSFORMS):
        assert sorted(DIR_PERM[t].tolist()) == [0, 1, 2, 3]


def test_bomb_and_wait_are_fixed_points():
    """Only the four moves rotate; BOMB and WAIT mean the same on any board."""
    for t in range(N_TRANSFORMS):
        for a in ('BOMB', 'WAIT'):
            assert transform_action(a, t) == a


def test_action_permutation_round_trips():
    for t in range(N_TRANSFORMS):
        for a in ACTIONS:
            assert inverse_transform_action(transform_action(a, t), t) == a


def test_action_index_and_name_agree():
    for t in range(N_TRANSFORMS):
        for i, a in enumerate(ACTIONS):
            assert transform_action(i, t) == ACTIONS.index(transform_action(a, t))
            assert ACTION_PERM[t][i] == ACTIONS.index(transform_action(a, t))


# --- the contract signature -----------------------------------------------

def test_apply_symmetry_returns_features_and_action():
    phi = state_to_features(make_state(BOARDS[0]))
    moved, action = apply_symmetry(phi, 'UP', transform_id=1)
    assert moved.shape == (FEATURE_DIM,)
    assert action in ACTIONS


def test_apply_symmetry_keeps_a_none_action_none():
    phi = state_to_features(make_state(BOARDS[0]))
    moved, action = apply_symmetry(phi, None, transform_id=3)
    assert action is None
    assert np.array_equal(moved, transform_features(phi, 3))


def test_apply_symmetry_rejects_an_out_of_range_transform():
    phi = state_to_features(make_state(BOARDS[0]))
    for bad in (-1, N_TRANSFORMS):
        try:
            apply_symmetry(phi, 'UP', transform_id=bad)
        except ValueError:
            continue
        raise AssertionError(f'transform_id={bad} should have raised')


# --- batch augmentation ---------------------------------------------------

def test_augment_batch_matches_the_single_state_path():
    phis = np.stack([state_to_features(make_state(p)) for p in BOARDS])
    actions = np.array([ACTIONS.index(a) for a in ('UP', 'LEFT', 'BOMB')])
    out_phi, out_act = augment_batch(phis, actions)

    n = len(phis)
    assert out_phi.shape == (N_TRANSFORMS * n, FEATURE_DIM)
    assert out_act.shape == (N_TRANSFORMS * n,)
    for t in range(N_TRANSFORMS):
        for i in range(n):
            assert np.array_equal(out_phi[t * n + i], transform_features(phis[i], t))
            assert out_act[t * n + i] == transform_action(int(actions[i]), t)


def test_augment_batch_keeps_states_and_next_states_aligned():
    """Two separate calls must put the same transform in the same block.

    If they did not, a state would be paired with a differently rotated
    successor and every TD target would be quietly wrong.
    """
    states = np.stack([state_to_features(make_state(p)) for p in BOARDS])
    next_states = states[::-1].copy()
    actions = np.zeros(len(states), dtype=np.intp)

    aug_states, _ = augment_batch(states, actions)
    aug_next, _ = augment_batch(next_states, actions)

    n = len(states)
    for t in range(N_TRANSFORMS):
        for i in range(n):
            row = t * n + i
            assert np.array_equal(aug_states[row], transform_features(states[i], t))
            assert np.array_equal(aug_next[row], transform_features(next_states[i], t))


# --- the form model.update() consumes -------------------------------------

def _sample_batch():
    phi = state_to_features(make_state(BOARDS[1]))
    nxt = state_to_features(make_state(BOARDS[0]))
    return phi, [Transition(phi, 'RIGHT', 1.0, nxt, False),
                 Transition(phi, 'BOMB', -2.0, None, True)]


def test_augment_transitions_expands_the_batch_eightfold():
    _, batch = _sample_batch()
    out = augment_transitions(batch)
    assert len(out) == N_TRANSFORMS * len(batch)
    for i, item in enumerate(batch):
        for t in range(N_TRANSFORMS):
            got = out[i * N_TRANSFORMS + t]
            assert np.array_equal(got.features, transform_features(item.features, t))
            assert got.action == transform_action(item.action, t)


def test_augment_transitions_carries_reward_and_done_and_keeps_terminals():
    """A rotated board pays the same reward and ends the same way."""
    _, batch = _sample_batch()
    for item in augment_transitions(batch):
        assert item.reward in (1.0, -2.0)
        assert item.done == (item.next_features is None)


def test_a_model_fitted_on_augmented_data_has_an_equivariant_q():
    """The payoff of augmentation: Q(T s, T a) == Q(s, a) for all eight T.

    The augmented training set is closed under the group, so the four
    directional weight blocks come out consistent with each other instead of
    each learning its own quirks. This is the property the report should show.
    """
    phi, batch = _sample_batch()
    # n_features passed explicitly rather than left to its default: the test
    # should fail if model.py's default ever stops tracking FEATURE_DIM.
    model = Model(n_features=FEATURE_DIM)
    model.update(augment_transitions(batch))

    q = model.predict_q(phi)
    for t in range(N_TRANSFORMS):
        q_t = model.predict_q(transform_features(phi, t))
        for i, action in enumerate(ACTIONS):
            j = ACTIONS.index(transform_action(action, t))
            assert np.isclose(q[i], q_t[j], atol=1e-5), \
                f'transform {t}, action {action}'
