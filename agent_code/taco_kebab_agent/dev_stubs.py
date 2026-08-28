"""Shared dev stubs for taco_kebab_agent (interface_contract.md, section 7).

This file lives at agent_code/taco_kebab_agent/dev_stubs.py and is committed
to the team's GitHub repository. All members import these stubs from here
(not from a personal or local-only copy) during parallel development.
"""

import numpy as np

from .features import FEATURE_DIM


def state_to_features_stub(game_state: dict) -> np.ndarray:
    """Temporary stand-in for Nehir's state_to_features (contract sections 2-3).

    Returns a random float32 vector of FEATURE_DIM, matching the shape and
    dtype of the real state_to_features. The size is imported rather than
    written out, so the stub cannot drift from the real vector.
    """
    return np.random.randn(FEATURE_DIM).astype(np.float32)


def reward_from_events_stub(events, old_game_state=None, new_game_state=None) -> float:
    """Temporary stand-in for Ege's reward_from_events (contract section 5).

    Placeholder, not meaningful — returns the number of events.
    """
    return float(len(events))  # placeholder, not meaningful
