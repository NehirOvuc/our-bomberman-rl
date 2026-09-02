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

from collections import deque

import numpy as np

from .callbacks import MODEL_PATH, state_to_features
from .model import Transition

# The real reward function (contract section 5). This replaced the
# dev_stubs placeholder on 28.08.2026 -- the single switch point Daniel
# left for exactly this swap.
from .rewards import reward_from_events

#: Raw transitions kept for refitting. Sized to hold an entire training run,
#: not a recent window: a 4000-round curriculum is about 211,000 steps, so the
#: first attempt at 20,000 held only the last 9.5% of it and never spanned more
#: than the final stage. That made it a hard-cutoff version of the forgetting
#: factor rather than a different mechanism, and it reproduced the same
#: amnesia. At this size nothing is evicted during a normal run, so the refit
#: below is genuine fitted Q-iteration: all the experience, none of the stale
#: targets. Roughly 57 MB of float32, which is training-time only -- the
#: submitted agent never allocates it.
REPLAY_SIZE = 250000

#: Rounds between refits. Every refit recomputes every target in the buffer
#: against the current Q-function and rebuilds the fit from scratch, so this
#: trades compute for freshness. 25 is a starting value, not a tuned one.
REFIT_EVERY = 25


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
    #: Raw experience: (features, action, reward, next_features, done). The
    #: reward stored here is the immediate reward, NOT a TD target -- targets
    #: are recomputed from scratch at every refit against the current
    #: Q-function, which is the whole point. A target computed in round 1 from
    #: a zero Q-function is simply wrong by round 4000, and the previous
    #: incremental version had no way to correct it.
    self.replay = deque(maxlen=REPLAY_SIZE)
    self.rounds_since_refit = 0
    self.round_reward = 0.0
    self.round_transition_count = 0


def _refit_from_replay(self):
    """Recompute every target against the current Q, then refit from scratch.

    This is one step of fitted Q-iteration. The targets are the same one-step
    TD targets the incremental version used --- reward + gamma * max_a Q(s', a),
    or just the reward on a terminal step --- so the only thing that has changed
    is when they are computed and that the fit does not carry old statistics.
    """
    rows = list(self.replay)
    if not rows:
        return

    rewards = np.array([r[2] for r in rows], dtype=np.float64)
    bootstrap = np.zeros(len(rows), dtype=np.float64)

    # Terminal steps have no successor and bootstrap nothing. The rest are
    # evaluated in one batched matrix product; doing it per transition would
    # make refitting too slow to run at this cadence.
    live = [i for i, r in enumerate(rows) if r[3] is not None]
    if live:
        next_features = np.stack([rows[i][3] for i in live])
        bootstrap[live] = self.model.predict_q_batch(next_features).max(axis=1)

    targets = rewards + self.model.gamma * bootstrap

    self.model.refit([
        Transition(features=r[0], action=r[1], reward=float(target),
                   next_features=r[3], done=r[4])
        for r, target in zip(rows, targets)
    ])
    self.rounds_since_refit = 0
    self.logger.info(f"Refit on {len(rows)} buffered transitions.")


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

    raw_reward = reward_from_events(events, old_game_state, new_game_state)
    self.round_reward += raw_reward

    self.replay.append((state_to_features(old_game_state), self_action, raw_reward,
                        state_to_features(new_game_state), False))
    self.round_transition_count += 1


def end_of_round(self, last_game_state: dict, last_action: str, events: list[str]):
    """
    Called at the end of each round to hand out final rewards and train on the round.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param last_game_state: The state that was passed to the last call of `act`.
    :param last_action: The action that was taken in response to `last_game_state`.
    :param events: The events that occurred during the agent's final step.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    raw_reward = reward_from_events(events, last_game_state, None)
    self.round_reward += raw_reward
    self.replay.append((state_to_features(last_game_state), last_action, raw_reward,
                        None, True))
    self.round_transition_count += 1

    self.rounds_since_refit += 1
    refitted = self.rounds_since_refit >= REFIT_EVERY
    if refitted:
        _refit_from_replay(self)

    # Save only when the model actually changed. On this branch neither model
    # learns between refits -- game_events_occurred just fills the buffer --
    # so a save on every other round writes an identical file. Free for Model
    # A's 9 KB .npz, not free for Model B: a full-size forest is 10.7 MB and
    # takes 0.23 s to compress, which is 16 minutes of pure I/O over a
    # 4000-round curriculum, longer than the training itself.
    if refitted:
        try:
            self.model.save(MODEL_PATH)
        except Exception as err:
            self.logger.error(f"Failed to save model to {MODEL_PATH}: {err}")

    self.logger.info(
        f"Round finished: total_reward={self.round_reward:.2f}, "
        f"transitions_processed={self.round_transition_count}, "
        f"replay={len(self.replay)}")

    self.round_reward = 0.0
    self.round_transition_count = 0
