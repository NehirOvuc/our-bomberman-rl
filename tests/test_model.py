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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_code.taco_kebab_agent.model import (  # noqa: E402
    ACTIONS, N_FEATURES, Model, Transition)


def q(model, features, action):
    """Look a single action's Q-value up by name instead of by index."""
    return model.predict_q(features)[ACTIONS.index(action)]


# --- contract ---------------------------------------------------------------

def test_predict_q_shape_and_dtype():
    model = Model()
    q_values = model.predict_q(np.zeros(N_FEATURES, dtype=np.float32))
    assert q_values.shape == (len(ACTIONS),)
    assert q_values.dtype == np.float32


def test_transition_field_order_matches_the_contract():
    t = Transition(features=1, action=2, reward=3, next_features=4, done=5)
    assert t._fields == ('features', 'action', 'reward', 'next_features', 'done')


def test_a_freshly_constructed_model_predicts_all_zeros():
    model = Model()
    assert np.all(model.predict_q(np.zeros(N_FEATURES, dtype=np.float32)) == 0.0)


def test_predict_q_rejects_the_wrong_feature_length():
    model = Model()
    try:
        model.predict_q(np.zeros(N_FEATURES - 1, dtype=np.float32))
        assert False, "expected ValueError"
    except ValueError as err:
        assert str(N_FEATURES) in str(err)


def test_update_rejects_the_wrong_feature_length():
    model = Model()
    bad_transition = Transition(
        features=np.zeros(N_FEATURES - 1, dtype=np.float32),
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
