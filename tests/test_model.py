"""Tests for model.py (Model A: ridge-regularised linear Q-learning).

Contract tests (shape, dtype, action order, relative-path enforcement)
protect everyone downstream: a `Model` that silently accepts the wrong
feature length breaks the ridge fit quietly instead of raising, and an
absolute save/load path breaks the Docker submission test. The rest check
that the ridge fit actually learns something and that actions don't leak
into each other's weights.

Run from the repository root:  python -m pytest tests
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_code.taco_kebab_agent.bfs import ACTIONS  # noqa: E402
from agent_code.taco_kebab_agent.features import FEATURE_DIM  # noqa: E402
from agent_code.taco_kebab_agent.model import Model, Transition  # noqa: E402


def q(model, features, action):
    """Look a single action's Q-value up by name instead of by index."""
    return model.predict_q(features)[ACTIONS.index(action)]


# --- contract ---------------------------------------------------------------

def test_predict_q_shape_and_dtype():
    model = Model()
    q_values = model.predict_q(np.zeros(FEATURE_DIM, dtype=np.float32))
    assert q_values.shape == (len(ACTIONS),)
    assert q_values.dtype == np.float32


def test_transition_field_order_matches_the_contract():
    t = Transition(features=1, action=2, reward=3, next_features=4, done=5)
    assert t._fields == ('features', 'action', 'reward', 'next_features', 'done')


def test_a_freshly_constructed_model_predicts_all_zeros():
    model = Model()
    assert np.all(model.predict_q(np.zeros(FEATURE_DIM, dtype=np.float32)) == 0.0)


def test_predict_q_rejects_the_wrong_feature_length():
    model = Model()
    try:
        model.predict_q(np.zeros(FEATURE_DIM - 1, dtype=np.float32))
        assert False, "expected ValueError"
    except ValueError as err:
        assert str(FEATURE_DIM) in str(err)


def test_update_rejects_the_wrong_feature_length():
    model = Model()
    bad_transition = Transition(
        features=np.zeros(FEATURE_DIM - 1, dtype=np.float32),
        action='WAIT', reward=0.0, next_features=None, done=True)
    try:
        model.update([bad_transition])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_save_and_load_reject_absolute_paths():
    model = Model()
    absolute = os.path.abspath(os.path.join('tests', '_absolute_model.npz'))
    try:
        model.save(absolute)
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        model.load(absolute)
        assert False, "expected ValueError"
    except ValueError:
        pass


# --- learning -----------------------------------------------------------

def test_update_only_changes_the_acted_action():
    model = Model(n_features=2)
    features = np.array([1.0, 0.0], dtype=np.float32)
    before = model.predict_q(features)

    model.update([Transition(features=features, action='UP', reward=5.0, next_features=None, done=True)])
    after = model.predict_q(features)

    assert after[ACTIONS.index('UP')] != before[ACTIONS.index('UP')]
    for action in ACTIONS:
        if action != 'UP':
            assert after[ACTIONS.index(action)] == before[ACTIONS.index(action)]


def test_ridge_fit_recovers_a_known_linear_target():
    """With ridge_lambda=0 and enough independent samples, the fit is exact OLS."""
    rng = np.random.default_rng(0)
    model = Model(n_features=3, ridge_lambda=0.0)
    weights, bias = np.array([2.0, -1.0, 0.5]), 0.25

    batch = []
    for _ in range(20):
        features = rng.normal(size=3).astype(np.float32)
        target = float(weights @ features + bias)
        batch.append(Transition(features=features, action='BOMB', reward=target, next_features=None, done=False))
    model.update(batch)

    probe = rng.normal(size=3).astype(np.float32)
    expected = float(weights @ probe + bias)
    assert abs(q(model, probe, 'BOMB') - expected) < 1e-3


def test_update_ignores_next_features_and_done():
    """The model regresses features -> reward only; next_features/done are the caller's business."""
    model = Model(n_features=2)
    features = np.array([1.0, 1.0], dtype=np.float32)
    with_terminal = Transition(features=features, action='WAIT', reward=1.0, next_features=None, done=True)
    without_terminal = Transition(features=features, action='LEFT', reward=1.0,
                                   next_features=np.zeros(2, dtype=np.float32), done=False)

    model.update([with_terminal])
    model.update([without_terminal])

    assert q(model, features, 'WAIT') == q(model, features, 'LEFT')


# --- persistence --------------------------------------------------------

def test_save_and_load_roundtrip():
    model = Model(n_features=2, ridge_lambda=0.5, n_step=3, gamma=0.9)
    features = np.array([1.0, -1.0], dtype=np.float32)
    model.update([Transition(features=features, action='DOWN', reward=2.0, next_features=None, done=True)])

    path = os.path.join('tests', '_roundtrip_model.npz')
    try:
        model.save(path)
        loaded = Model(n_features=2)
        loaded.load(path)

        assert np.array_equal(model.predict_q(features), loaded.predict_q(features))
        assert loaded.ridge_lambda == 0.5
        assert loaded.n_step == 3
        assert loaded.gamma == 0.9
    finally:
        if os.path.exists(path):
            os.remove(path)


# --- forgetting factor -----------------------------------------------------

def test_forget_defaults_to_keeping_everything():
    # The old behaviour, kept as the default so nothing changes unasked.
    assert Model().forget == 1.0


def test_accumulated_statistics_grow_without_bound_when_nothing_is_forgotten():
    # This is the bug, pinned. With forget = 1 the fit statistics grow with
    # every batch, so the influence of each new batch shrinks as a run goes on.
    m = Model(forget=1.0)
    batch = [Transition(features=np.ones(FEATURE_DIM, dtype=np.float32),
                        action='BOMB', reward=1.0, next_features=None, done=True)]
    norms = []
    for _ in range(50):
        m.update(batch)
        norms.append(np.linalg.norm(m._A['BOMB']))
    assert norms[-1] > 5 * norms[0]


def test_forgetting_keeps_the_statistics_bounded():
    m = Model(forget=0.9)
    batch = [Transition(features=np.ones(FEATURE_DIM, dtype=np.float32),
                        action='BOMB', reward=1.0, next_features=None, done=True)]
    norms = []
    for _ in range(200):
        m.update(batch)
        norms.append(np.linalg.norm(m._A['BOMB']))
    # Settles rather than growing: the last stretch barely moves.
    assert abs(norms[-1] - norms[-50]) < 0.01 * norms[-1]


def test_recent_targets_win_when_the_target_changes():
    # The property the change exists for. Fit hard on one target, then feed a
    # different one repeatedly; a forgetting model must follow the new target
    # while a remembering one stays anchored to the old.
    phi = np.ones(FEATURE_DIM, dtype=np.float32)
    old = [Transition(features=phi, action='BOMB', reward=10.0,
                      next_features=None, done=True)]
    new = [Transition(features=phi, action='BOMB', reward=-10.0,
                      next_features=None, done=True)]

    remembering, forgetting = Model(forget=1.0), Model(forget=0.9)
    for m in (remembering, forgetting):
        for _ in range(100):
            m.update(old)
        for _ in range(100):
            m.update(new)

    q_remember = float(remembering.predict_q(phi)[ACTIONS.index('BOMB')])
    q_forget = float(forgetting.predict_q(phi)[ACTIONS.index('BOMB')])
    assert q_forget < q_remember
    assert q_forget < 0        # the forgetting model has followed the new target


def test_forget_survives_save_and_load(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m = Model(forget=0.95)
    m.save('m.npz')
    loaded = Model()
    loaded.load('m.npz')
    assert loaded.forget == pytest.approx(0.95)


# --- replay refit ----------------------------------------------------------

def test_batched_prediction_matches_one_at_a_time():
    m = Model()
    m.update([Transition(features=np.ones(FEATURE_DIM, dtype=np.float32), action='UP',
                         reward=3.0, next_features=None, done=True)])
    states = np.random.RandomState(0).rand(7, FEATURE_DIM).astype(np.float32)
    batched = m.predict_q_batch(states)
    one_by_one = np.stack([m.predict_q(s) for s in states])
    assert np.allclose(batched, one_by_one, atol=1e-5)


def test_refit_forgets_the_old_target_completely():
    # The property the replay buffer exists for, and the one the forgetting
    # factor could only approximate: after a refit, data that is not in the
    # batch has no influence at all.
    phi = np.ones(FEATURE_DIM, dtype=np.float32)
    m = Model()
    for _ in range(50):
        m.update([Transition(features=phi, action='BOMB', reward=10.0,
                             next_features=None, done=True)])
    assert m.predict_q(phi)[ACTIONS.index('BOMB')] > 5

    m.refit([Transition(features=phi, action='BOMB', reward=-10.0,
                        next_features=None, done=True)])
    assert m.predict_q(phi)[ACTIONS.index('BOMB')] < 0


def test_refit_does_not_let_the_statistics_grow():
    m = Model()
    batch = [Transition(features=np.ones(FEATURE_DIM, dtype=np.float32), action='BOMB',
                        reward=1.0, next_features=None, done=True)]
    m.refit(batch)
    first = np.linalg.norm(m._A['BOMB'])
    for _ in range(100):
        m.refit(batch)
    assert np.linalg.norm(m._A['BOMB']) == pytest.approx(first)


def test_refit_leaves_the_ridge_penalty_in_place():
    # Refitting resets to the penalty, not to zero, or a sparsely-visited
    # action would give a singular matrix on the next solve.
    m = Model()
    m.refit([])
    assert np.linalg.norm(m._A['WAIT']) > 0
