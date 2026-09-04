"""Tests for model_b.py (Model B: fitted Q-iteration with a regression forest).

Two jobs. First, the contract tests that protect everyone downstream --
shape, dtype, action order, relative-path enforcement, round-tripping through
joblib -- mirrored from test_model.py so that A and B are held to the same
interface and `train.py` can drive either without knowing which it has.

Second, and the reason Model B exists at all: the conjunction test at the
bottom, which checks that a forest fits "bombing pays only when a crate is in
the blast AND an escape route exists" far more accurately than a hyperplane
can. Writing it turned up that our stated version of that claim was too strong
-- see the test's own docstring. Better to find that here than in the report.

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
from agent_code.taco_kebab_agent.model_b import ModelB  # noqa: E402

#: Feature indices this test file leans on, by name rather than by number, so
#: the tests say what they mean and break loudly if features.py is reordered.
BOMB_CRATE_COUNT = 30
BOMB_ESCAPE_POSSIBLE = 32


def _transitions(rng, n, action, reward_fn):
    """n transitions for one action, with features drawn uniformly at random."""
    rows = []
    for _ in range(n):
        phi = rng.random(FEATURE_DIM).astype(np.float32)
        rows.append(Transition(features=phi, action=action,
                               reward=float(reward_fn(phi)),
                               next_features=None, done=True))
    return rows


# --------------------------------------------------------------------------
# Interface parity with Model A
# --------------------------------------------------------------------------

def test_predict_q_shape_and_dtype_before_any_fit():
    """An unfitted model must still answer, so act() works on round one."""
    model = ModelB()
    q = model.predict_q(np.zeros(FEATURE_DIM, dtype=np.float32))
    assert q.shape == (len(ACTIONS),)
    assert q.dtype == np.float32
    assert np.all(q == 0.0)
    assert not model.is_fitted


def test_predict_q_rejects_wrong_feature_length():
    """Silently accepting the wrong length would corrupt the fit, not raise."""
    model = ModelB()
    with pytest.raises(ValueError):
        model.predict_q(np.zeros(FEATURE_DIM + 1, dtype=np.float32))


def test_predict_q_batch_matches_predict_q_row_by_row():
    """The refit uses the batched path; it must agree with the one act() uses."""
    rng = np.random.default_rng(0)
    model = ModelB(n_estimators=10)
    model.refit(_transitions(rng, 400, 'BOMB', lambda p: p[BOMB_CRATE_COUNT]))

    states = rng.random((16, FEATURE_DIM)).astype(np.float32)
    batch = model.predict_q_batch(states)
    assert batch.shape == (16, len(ACTIONS))
    for i, state in enumerate(states):
        np.testing.assert_allclose(batch[i], model.predict_q(state), rtol=1e-5)


def test_predict_q_batch_rejects_wrong_shape():
    model = ModelB()
    with pytest.raises(ValueError):
        model.predict_q_batch(np.zeros((4, FEATURE_DIM + 1), dtype=np.float32))


def test_action_order_matches_the_contract():
    """Index i must mean ACTIONS[i] in both models, or act() picks the wrong one."""
    assert ModelB().actions == ACTIONS == Model().actions


def test_update_refuses_rather_than_fitting_on_one_batch():
    """The one deliberate deviation from the contract, asserted so it stays loud.

    Fitting a forest on `batch` alone would discard every earlier transition
    and return a model trained on the last round -- wrong, with nothing to
    notice. Model B is refit-only by design (PLAN.md section 2).
    """
    with pytest.raises(NotImplementedError, match="refit"):
        ModelB().update([])


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------

def test_refit_learns_a_signal_it_was_shown():
    rng = np.random.default_rng(1)
    model = ModelB(n_estimators=20)
    model.refit(_transitions(rng, 1000, 'BOMB', lambda p: 5.0 * p[BOMB_CRATE_COUNT]))

    high = np.zeros(FEATURE_DIM, dtype=np.float32); high[BOMB_CRATE_COUNT] = 1.0
    low = np.zeros(FEATURE_DIM, dtype=np.float32); low[BOMB_CRATE_COUNT] = 0.0
    bomb = ACTIONS.index('BOMB')
    assert model.predict_q(high)[bomb] > model.predict_q(low)[bomb]


def test_actions_do_not_leak_into_each_other():
    """Six independent regressors: training BOMB must not move UP's Q."""
    rng = np.random.default_rng(2)
    model = ModelB(n_estimators=10)
    model.refit(_transitions(rng, 500, 'BOMB', lambda p: 3.0))

    state = rng.random(FEATURE_DIM).astype(np.float32)
    q = model.predict_q(state)
    assert q[ACTIONS.index('BOMB')] == pytest.approx(3.0, abs=0.5)
    assert q[ACTIONS.index('UP')] == 0.0


def test_an_action_below_the_sample_floor_is_not_fit_on_noise():
    """BOMB is rare early; a forest on 10 rows is worse than an honest zero."""
    rng = np.random.default_rng(3)
    model = ModelB(n_estimators=10)
    model.refit(_transitions(rng, 10, 'BOMB', lambda p: 3.0))
    assert not model.is_fitted
    assert np.all(model.predict_q(rng.random(FEATURE_DIM).astype(np.float32)) == 0.0)


def test_refit_replaces_rather_than_accumulates():
    """Fitted Q-iteration: the second refit must not remember the first."""
    rng = np.random.default_rng(4)
    model = ModelB(n_estimators=10)
    state = rng.random(FEATURE_DIM).astype(np.float32)
    bomb = ACTIONS.index('BOMB')

    model.refit(_transitions(rng, 500, 'BOMB', lambda p: 10.0))
    assert model.predict_q(state)[bomb] == pytest.approx(10.0, abs=1.0)
    model.refit(_transitions(rng, 500, 'BOMB', lambda p: -10.0))
    assert model.predict_q(state)[bomb] == pytest.approx(-10.0, abs=1.0)


# --------------------------------------------------------------------------
# Save / load
# --------------------------------------------------------------------------

def test_save_and_load_reject_absolute_paths():
    """Absolute paths break the Docker submission test."""
    model = ModelB()
    with pytest.raises(ValueError):
        model.save('/tmp/model_b.joblib')
    with pytest.raises(ValueError):
        model.load('/tmp/model_b.joblib')


def test_round_trip_through_joblib_preserves_predictions(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    model = ModelB(n_estimators=10)
    model.refit(_transitions(rng, 600, 'BOMB', lambda p: 4.0 * p[BOMB_CRATE_COUNT]))
    states = rng.random((8, FEATURE_DIM)).astype(np.float32)
    before = model.predict_q_batch(states)

    monkeypatch.chdir(tmp_path)   # save() takes a relative path, so supply a cwd
    model.save('model_b.joblib')
    reloaded = ModelB()
    reloaded.load('model_b.joblib')

    np.testing.assert_allclose(reloaded.predict_q_batch(states), before, rtol=1e-6)
    assert reloaded.gamma == model.gamma


def test_a_loaded_model_never_predicts_in_parallel(tmp_path, monkeypatch):
    """"No multiprocessing in the final agent" enforced by construction.

    Fitting may use every core -- that is training-time. Prediction must not,
    in this process or in the tournament process that loads the file.
    """
    rng = np.random.default_rng(6)
    model = ModelB(n_estimators=10, n_jobs=-1)
    model.refit(_transitions(rng, 500, 'BOMB', lambda p: 1.0))
    assert model._forests['BOMB'].n_jobs == 1

    monkeypatch.chdir(tmp_path)
    model.save('model_b.joblib')
    reloaded = ModelB()
    reloaded.load('model_b.joblib')
    assert reloaded._forests['BOMB'].n_jobs == 1


# --------------------------------------------------------------------------
# The reason Model B exists
# --------------------------------------------------------------------------

def test_forest_fits_a_conjunction_the_linear_model_can_only_approximate():
    """The hypothesis behind this whole branch, stated as narrowly as it is true.

    "Bombing pays" is a conjunction: a crate is in the blast AND an escape
    route exists. Reward +1 when both hold, -1 otherwise.

    Note what this does NOT assert, because measuring it showed the stronger
    claim is false: a hyperplane with a positive weight on each feature ranks
    the (1,1) corner top perfectly well, since Q(1,1) - max(Q(1,0), Q(0,1)) =
    min(w_crate, w_escape) > 0. Linear models represent AND-of-positives fine.

    What it cannot do is get the *shape* right. It has to interpolate, so it
    ramps where the truth steps, and it is wrong precisely in the two corners
    that matter operationally -- crate but no escape, escape but no crate --
    the states where dropping a bomb is a mistake. Measured over 4000 random
    states, uniform across the five seeds tried: linear MSE ~0.38, forest MSE
    ~0.027, a ~14x gap, and 91% against 99.7% agreement on the sign of Q.

    That is the honest version of the claim section 6 should make.
    """
    rng = np.random.default_rng(7)

    def payoff(phi):
        return 1.0 if phi[BOMB_CRATE_COUNT] > 0.5 and phi[BOMB_ESCAPE_POSSIBLE] > 0.5 else -1.0

    batch = _transitions(rng, 3000, 'BOMB', payoff)

    linear = Model()
    linear.update(batch)
    forest = ModelB(n_estimators=30)
    forest.refit(batch)

    bomb = ACTIONS.index('BOMB')

    def corner(crate, escape):
        # Every other feature at 0.5 so only the two under test vary.
        phi = np.full(FEATURE_DIM, 0.5, dtype=np.float32)
        phi[BOMB_CRATE_COUNT] = crate
        phi[BOMB_ESCAPE_POSSIBLE] = escape
        return phi

    # The forest gets all four corners about right.
    assert forest.predict_q(corner(1.0, 1.0))[bomb] > 0.5
    for phi in (corner(0.0, 0.0), corner(1.0, 0.0), corner(0.0, 1.0)):
        assert forest.predict_q(phi)[bomb] < 0.0

    # The linear model does not: it is dragged well above the true -1.0 in the
    # two half-corners, which are exactly the states where a bomb kills us.
    for phi in (corner(1.0, 0.0), corner(0.0, 1.0)):
        assert linear.predict_q(phi)[bomb] > -1.0

    # The summary claim: an order of magnitude less approximation error, on
    # the same data, with only the approximator changed.
    states = rng.random((4000, FEATURE_DIM)).astype(np.float32)
    truth = np.array([payoff(s) for s in states])
    q_linear = np.array([linear.predict_q(s)[bomb] for s in states])
    q_forest = forest.predict_q_batch(states)[:, bomb]

    mse_linear = float(((q_linear - truth) ** 2).mean())
    mse_forest = float(((q_forest - truth) ** 2).mean())
    assert mse_forest < mse_linear / 5, f"linear {mse_linear:.3f}, forest {mse_forest:.3f}"
    assert (np.sign(q_forest) == np.sign(truth)).mean() > 0.98
    assert (np.sign(q_linear) == np.sign(truth)).mean() < 0.95
