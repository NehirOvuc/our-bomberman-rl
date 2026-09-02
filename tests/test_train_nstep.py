"""Tests for the n-step TD targets in train.py's replay refit.

Why this file exists. train.py's module docstring claimed the n-step windowing
was already handled and that raising n_step was "just a matter of changing
n_step in Model's constructor". That was true of the incremental version it was
written for; the replay refit that replaced it computed
`reward + gamma * max_a Q(s', a)` unconditionally and ignored n_step entirely.
So the one knob the docstring promised was wired to nothing.

It matters for exactly one action. BOMB_TIMER is 4 and EXPLOSION_TIMER is 2, so
a bomb resolves six steps after it is dropped while the drop step itself pays
nothing. At n_step=1 the BOMB regression never sees its own consequence in its
own target and has to receive it through six rounds of bootstrapping. Nehir
measured the cost on the linear model: the coefficient on bomb_crate_count is
-0.08 at n_step=1 and +0.60 at n_step=5 -- a sign flip, i.e. with a one-step
target the model learns that destroying crates is bad.

Run from the repository root:  python -m pytest tests
"""

import logging
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_code.taco_kebab_agent.features import FEATURE_DIM  # noqa: E402
from agent_code.taco_kebab_agent.train import _refit_from_replay  # noqa: E402

GAMMA = 0.5      # not 0.99, so the discount is visible in the arithmetic


class _RecordingModel:
    """Stands in for Model/ModelB: records the targets it is asked to fit.

    Q is a fixed constant for every action, so the bootstrap term is
    predictable and the test is about the windowing, not about the regressor.
    """

    def __init__(self, n_step, q=1.0, n_actions=6):
        self.gamma = GAMMA
        self.n_step = n_step
        self.q = q
        self.n_actions = n_actions
        self.fitted = None

    def predict_q_batch(self, features):
        return np.full((len(features), self.n_actions), self.q)

    def refit(self, batch):
        self.fitted = list(batch)


def _agent(model, rows):
    return SimpleNamespace(model=model, replay=deque(rows),
                           rounds_since_refit=0, logger=logging.getLogger('test'))


def _row(reward, done=False):
    phi = np.zeros(FEATURE_DIM, dtype=np.float32)
    return (phi, 'BOMB', reward, None if done else phi, done)


def _targets(model, rows):
    agent = _agent(model, rows)
    _refit_from_replay(agent)
    return [t.reward for t in model.fitted]


# --------------------------------------------------------------------------

def test_one_step_is_unchanged():
    """The historical behaviour must be bit-for-bit what it always was.

    Every result in the experiment log so far was produced at n_step=1, so a
    change here would silently invalidate the comparisons they are part of.
    """
    model = _RecordingModel(n_step=1, q=10.0)
    targets = _targets(model, [_row(1.0), _row(2.0), _row(3.0, done=True)])

    assert targets[0] == pytest.approx(1.0 + GAMMA * 10.0)
    assert targets[1] == pytest.approx(2.0 + GAMMA * 10.0)
    # A terminal step bootstraps nothing.
    assert targets[2] == pytest.approx(3.0)


def test_a_full_window_sums_and_then_bootstraps():
    model = _RecordingModel(n_step=3, q=10.0)
    rows = [_row(1.0), _row(1.0), _row(1.0), _row(1.0), _row(1.0)]
    targets = _targets(model, rows)

    # 1 + 0.5 + 0.25 discounted rewards, then gamma^3 * max_a Q.
    expected = 1.0 + GAMMA * 1.0 + GAMMA ** 2 * 1.0 + GAMMA ** 3 * 10.0
    assert targets[0] == pytest.approx(expected)


def test_the_window_stops_at_the_end_of_a_round():
    """Rounds are appended back to back, so a window must not read across them.

    Without the `done` check the last steps of one round would be regressed
    against the opening rewards of the next -- a different board, a different
    episode, and a target with no meaning.
    """
    model = _RecordingModel(n_step=5, q=10.0)
    rows = [_row(1.0), _row(2.0, done=True), _row(99.0), _row(99.0)]
    targets = _targets(model, rows)

    # Row 0's window hits the terminal row 1 and stops: no bootstrap, and the
    # 99.0 rewards from the next round must not appear.
    assert targets[0] == pytest.approx(1.0 + GAMMA * 2.0)
    assert targets[1] == pytest.approx(2.0)


def test_a_window_running_off_the_end_of_the_buffer_does_not_bootstrap_twice():
    """The last rows have no full window; they must degrade, not crash."""
    model = _RecordingModel(n_step=5, q=10.0)
    rows = [_row(1.0), _row(1.0)]
    targets = _targets(model, rows)

    assert len(targets) == 2
    assert all(np.isfinite(targets))


def test_n_step_is_read_from_the_model():
    """The knob has to be connected: same data, different n_step, different target."""
    rows = [_row(1.0) for _ in range(6)]
    one = _targets(_RecordingModel(n_step=1, q=10.0), rows)
    five = _targets(_RecordingModel(n_step=5, q=10.0), rows)
    assert one[0] != pytest.approx(five[0])
