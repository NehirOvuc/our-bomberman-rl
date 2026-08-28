"""Training callbacks for taco_kebab_agent (interface_contract.md 'Reference' section).

Owns the training-only loop: raw per-step data -> n-step TD targets ->
Model.update. state_to_features is reused from callbacks.py's own single
switch point (dev_stubs.state_to_features_stub for now) rather than
imported a second time here.

Design decision: this first version relies on Model's own default of
n_step=1, which collapses the general n-step windowing below to plain
1-step TD: target = raw_reward + gamma * max_q(next_features). Generalizing
to n>1 later is just a matter of changing n_step in Model's constructor --
the windowing logic in game_events_occurred/end_of_round already handles
it, no rewrite needed.
"""

import numpy as np

from .callbacks import MODEL_PATH, state_to_features
from .model import Transition

# The real reward function (contract section 5). This replaced the
# dev_stubs placeholder on 28.08.2026 -- the single switch point Daniel
# left for exactly this swap.
from .rewards import reward_from_events

#: Transitions accumulated before an online Model.update call mid-round.
BATCH_SIZE = 32


def _n_step_return(rewards, gamma, bootstrap=None):
    """Discounted sum of `rewards`, optionally plus a discounted bootstrap term.

    With `bootstrap=None` this is the plain (truncated) Monte-Carlo return
    used to flush the buffer at the end of a round. With a bootstrap value
    it is the standard n-step TD target used mid-round.
    """
    target = sum(gamma ** k * r for k, r in enumerate(rewards))
    if bootstrap is not None:
        target += gamma ** len(rewards) * bootstrap
    return target


def setup_training(self):
    """
    Initialise self for training purpose. Called once, after setup() in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.step_buffer = []          # raw per-step tuples, before n-step-return computation
    self.pending_transitions = []  # fully-formed Transitions, ready for Model.update
    self.round_reward = 0.0
    self.round_transition_count = 0


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: list[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from `old_game_state` to `new_game_state`.
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    features = state_to_features(old_game_state)
    next_features = state_to_features(new_game_state)
    raw_reward = reward_from_events(events, old_game_state, new_game_state)
    self.round_reward += raw_reward

    self.step_buffer.append((features, self_action, raw_reward, next_features, False))

    n_step = self.model.n_step
    if len(self.step_buffer) >= n_step:
        oldest = self.step_buffer.pop(0)
        window = [oldest] + self.step_buffer[:n_step - 1]
        rewards = [entry[2] for entry in window]
        bootstrap_features = window[-1][3]  # state n_step steps after `oldest`
        bootstrap = float(np.max(self.model.predict_q(bootstrap_features)))
        target = _n_step_return(rewards, self.model.gamma, bootstrap)

        self.pending_transitions.append(Transition(
            features=oldest[0], action=oldest[1], reward=target,
            next_features=oldest[3], done=False))
        self.round_transition_count += 1

    if len(self.pending_transitions) >= BATCH_SIZE:
        self.model.update(self.pending_transitions)
        self.pending_transitions = []


def end_of_round(self, last_game_state: dict, last_action: str, events: list[str]):
    """
    Called at the end of each round to hand out final rewards and train on the round.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param last_game_state: The state that was passed to the last call of `act`.
    :param last_action: The action that was taken in response to `last_game_state`.
    :param events: The events that occurred during the agent's final step.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    features = state_to_features(last_game_state)
    raw_reward = reward_from_events(events, last_game_state, None)
    self.round_reward += raw_reward
    self.step_buffer.append((features, last_action, raw_reward, None, True))

    # Flush the whole buffer: every remaining entry gets the actual return to
    # the end of the episode, no bootstrap -- there is no future Q-value left
    # to estimate once the round is over.
    rewards = [entry[2] for entry in self.step_buffer]
    for i, entry in enumerate(self.step_buffer):
        target = _n_step_return(rewards[i:], self.model.gamma)
        self.pending_transitions.append(Transition(
            features=entry[0], action=entry[1], reward=target,
            next_features=entry[3], done=entry[4]))
        self.round_transition_count += 1

    self.model.update(self.pending_transitions)

    try:
        self.model.save(MODEL_PATH)
    except Exception as err:
        self.logger.error(f"Failed to save model to {MODEL_PATH}: {err}")

    self.logger.info(
        f"Round finished: total_reward={self.round_reward:.2f}, "
        f"transitions_processed={self.round_transition_count}")

    self.step_buffer = []
    self.pending_transitions = []
    self.round_reward = 0.0
    self.round_transition_count = 0
