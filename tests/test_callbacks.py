"""Tests for callbacks.py's agent-owned exploration RNG (TACO_SEED).

The framework's own --seed flag deliberately does not cover agent
randomness (only world generation -- crate/coin placement; see the project
PDF), so act()'s epsilon-greedy exploration used the global `np.random` state
until this: two nominally identical training runs diverged in every
exploration decision from the first step, with no way to reproduce a run.
TACO_SEED gives each agent its own `np.random.default_rng`, following the
same env-var convention as TACO_MODEL / TACO_NSTEP / TACO_FRESH.

Run from the repository root:  python -m pytest tests
"""

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helpers import make_state  # noqa: E402

from agent_code.taco_kebab_agent import callbacks  # noqa: E402
from agent_code.taco_kebab_agent.model import ACTIONS  # noqa: E402

STATE = make_state("""
#####
#a..#
#####
""")


def _agent(train=True):
    return SimpleNamespace(train=train, logger=logging.getLogger('test'))


def test_same_seed_gives_identical_rng_draws(monkeypatch):
    """Two agents set up with the same TACO_SEED must draw the same
    random() values and the same choice() picks from their own rng -- the
    exact mechanism act() relies on for exploration.
    """
    monkeypatch.setattr(callbacks, 'AGENT_SEED', '12345')
    agent1, agent2 = _agent(), _agent()
    callbacks.setup(agent1)
    callbacks.setup(agent2)

    draws1 = [agent1.rng.random() for _ in range(20)]
    draws2 = [agent2.rng.random() for _ in range(20)]
    assert draws1 == draws2

    picks1 = [agent1.rng.choice(ACTIONS) for _ in range(20)]
    picks2 = [agent2.rng.choice(ACTIONS) for _ in range(20)]
    assert picks1 == picks2


def test_same_seed_gives_identical_exploration_decisions_through_act(monkeypatch):
    """End-to-end: same TACO_SEED -> the same sequence of actions from act().

    act() draws exactly one random() and, when it explores, one choice() per
    call; a freshly-initialised Model predicts the same (zero) Q-values for
    both agents here, so any divergence in the returned action sequence can
    only come from the agents' rng instances disagreeing.
    """
    monkeypatch.setattr(callbacks, 'AGENT_SEED', '999')
    agent1, agent2 = _agent(), _agent()
    callbacks.setup(agent1)
    callbacks.setup(agent2)

    actions1 = [callbacks.act(agent1, STATE) for _ in range(30)]
    actions2 = [callbacks.act(agent2, STATE) for _ in range(30)]
    assert actions1 == actions2
    # Sanity check that this exercised both branches of act(), or the
    # comparison above would pass vacuously (e.g. if exploration never
    # triggered for this seed, this would only prove the exploit branch --
    # which does not even touch the rng -- is deterministic).
    assert len(set(actions1)) > 1


def test_unset_seed_does_not_raise_and_still_returns_valid_actions(monkeypatch):
    """TACO_SEED left unset must not change the *structure* of setup()/act():
    no exception, and act() still returns a legal action. Reproducibility is
    not expected here -- default_rng(None) seeds from OS entropy, on purpose,
    to preserve today's unseeded behaviour.
    """
    monkeypatch.setattr(callbacks, 'AGENT_SEED', None)
    agent = _agent()
    callbacks.setup(agent)

    assert isinstance(agent.rng, np.random.Generator)
    for _ in range(10):
        action = callbacks.act(agent, STATE)
        assert action in ACTIONS


def test_malformed_seed_fails_at_setup_not_at_import(monkeypatch):
    """AGENT_SEED is read as a string at module scope and only parsed to int
    inside setup(), so a malformed TACO_SEED must fail there -- loudly --
    rather than being silently swallowed at import time.
    """
    monkeypatch.setattr(callbacks, 'AGENT_SEED', 'not-an-int')
    with pytest.raises(ValueError):
        callbacks.setup(_agent())
