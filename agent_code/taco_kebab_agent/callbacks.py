"""Callback interface for taco_kebab_agent (interface_contract.md, 'Reference' section).

Owns state -> action only: state_to_features -> Model.predict_q -> pick action.
No training logic lives here -- game_events_occurred / end_of_round /
Model.update live in train.py.
"""

import os
import random

import numpy as np

from .model import ACTIONS, Model
from .model_b import ModelB

from .features import state_to_features

#: Which approximator to build. Set TACO_MODEL=b to run the forest, mirroring
#: the TACO_FRESH switch below. An environment variable rather than two agent
#: directories on purpose: it is the only way to guarantee the A and B arms
#: share one act(), one feature call and one epsilon, which is what makes
#: section 6 a controlled comparison instead of two separate agents.
MODEL_KIND = os.environ.get('TACO_MODEL', 'a').lower()

#: TD window length. 1 is the historical default. BOMB_TIMER (4) plus
#: EXPLOSION_TIMER (2) is 6, so a bomb resolves six steps after it is dropped
#: while the drop step itself pays nothing -- at n_step=1 the BOMB regression
#: never sees its own consequence directly. Set TACO_NSTEP=5 to span the fuse.
N_STEP = int(os.environ.get('TACO_NSTEP', '1'))

#: Relative path per interface_contract.md section 6 -- absolute paths break
#: the Docker submission test. Bare filename because SequentialAgentBackend
#: (agents.py) chdirs into this agent's own directory before every callback,
#: so this is already relative to agent_code/taco_kebab_agent/.
#: Model A saves .npz, Model B .joblib, so the two arms cannot overwrite each
#: other's weights when they are trained back to back.
#:
#: Set TACO_MODEL_PATH to override this computed default and point at a
#: specific saved file instead. tools/evaluate.py's --lineup mirror needs
#: this to compare two saved model versions against each other, which
#: otherwise requires manually copying the whole agent directory, since
#: MODEL_PATH is otherwise fixed. Same bare-filename constraint as above
#: applies, and it's validated eagerly here (like N_STEP) rather than
#: deferred to setup() the way AGENT_SEED is: a malformed override is a
#: config mistake to catch immediately, not a training input to react to.
TACO_MODEL_PATH = os.environ.get('TACO_MODEL_PATH')
if TACO_MODEL_PATH is not None and os.path.basename(TACO_MODEL_PATH) != TACO_MODEL_PATH:
    raise ValueError(
        f"TACO_MODEL_PATH must be a bare filename with no directory "
        f"component -- got {TACO_MODEL_PATH!r}."
    )

MODEL_PATH = TACO_MODEL_PATH or ('model_b.joblib' if MODEL_KIND == 'b' else 'model_a.npz')

#: Seed for this agent's own exploration RNG. The framework's --seed flag
#: deliberately does not cover agent randomness (only world generation --
#: crate/coin placement), so two nominally identical training runs otherwise
#: diverge in every epsilon-greedy decision from the first step. Left
#: unparsed here -- a malformed TACO_SEED should fail loudly in setup() where
#: it's used, not silently at import time. Unset by default, which preserves
#: today's unseeded behaviour.
AGENT_SEED = os.environ.get('TACO_SEED')


def setup(self):
    """
    Set up the model for this agent. Called once, before act() is ever called.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # forget stays at its default of 1.0 on this branch: train.py refits from
    # a replay buffer instead, which throws the old statistics away wholesale
    # rather than fading them, so a discount on top would be a second
    # mechanism doing the same job less well.
    #
    # Model B is the same pipeline with the linear Q swapped for a forest.
    # Nothing else differs -- same features, same rewards, same epsilon.
    self.model = (ModelB(n_step=N_STEP) if MODEL_KIND == 'b'
                  else Model(n_step=N_STEP))
    self.logger.info(f"Using model {MODEL_KIND.upper()} with weights at "
                     f"{MODEL_PATH}, n_step={N_STEP}.")
    self.epsilon = 0.2  # exploration rate; tuned later via PLAN.md's hyperparameter grid search

    # default_rng(None) would seed from OS entropy, so no --seed could make
    # a run repeatable: this Generator is independent of numpy's global
    # state, and it drives the epsilon-greedy draw and the tie-break in
    # act() -- the tie-break unconditionally, so evaluation was as
    # unrepeatable as training. Draw the seed from the stdlib generator,
    # which main.py seeds from --seed and which no agent reseeds; numpy's
    # global is unusable here because several framework agents call
    # np.random.seed() with no argument in their own setup(). Unseeded
    # runs behave as before, and TACO_SEED still overrides both.
    seed = int(AGENT_SEED) if AGENT_SEED is not None else random.getrandbits(32)
    self.rng = np.random.default_rng(seed)

    # Training continues from the saved weights when there are any. The
    # previous version returned here before the load whenever self.train was
    # set, so every training run started from zero -- which made the staged
    # curriculum in training_scenarios.py impossible to run at all: there was
    # no way to train on coin-heaven and carry the weights into crate-easy.
    #
    # Set TACO_FRESH=1 in the environment to start from scratch on purpose,
    # which is what a from-zero baseline needs.
    if os.environ.get('TACO_FRESH') == '1':
        self.logger.info("TACO_FRESH set: starting from a fresh, untrained model.")
        return

    if os.path.isfile(MODEL_PATH):
        self.logger.info(f"Loading model from {MODEL_PATH}.")
        self.model.load(MODEL_PATH)
    elif self.train:
        self.logger.info("Training mode: no saved model yet, starting fresh.")
    else:
        self.logger.error(f"No trained model found at {MODEL_PATH}; playing with an untrained model.")


def act(self, game_state: dict) -> str:
    """
    Decide on an action given the current game state.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    features = state_to_features(game_state)
    q_values = self.model.predict_q(features)

    if self.train and self.rng.random() < self.epsilon:
        action = self.rng.choice(ACTIONS)
        self.logger.debug(f"Exploring: chose random action {action}.")
        return action

    # np.argmax always resolves a tie to its first index, which is UP -- and
    # every action is tied at 0.0 before the model has seen any data, so an
    # untrained agent used to open every round by walking into a wall. Break
    # ties uniformly instead, over all actions sharing the max, not just two.
    best = np.flatnonzero(q_values == q_values.max())
    action = ACTIONS[self.rng.choice(best)]
    self.logger.debug(f"Exploiting: chose action {action} from Q-values {q_values}.")
    return action
