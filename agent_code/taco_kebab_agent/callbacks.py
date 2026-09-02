"""Callback interface for taco_kebab_agent (interface_contract.md, 'Reference' section).

Owns state -> action only: state_to_features -> Model.predict_q -> pick action.
No training logic lives here -- game_events_occurred / end_of_round /
Model.update live in train.py.
"""

import os

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

#: Relative path per interface_contract.md section 6 -- absolute paths break
#: the Docker submission test. Bare filename because SequentialAgentBackend
#: (agents.py) chdirs into this agent's own directory before every callback,
#: so this is already relative to agent_code/taco_kebab_agent/.
#: Model A saves .npz, Model B .joblib, so the two arms cannot overwrite each
#: other's weights when they are trained back to back.
MODEL_PATH = 'model_b.joblib' if MODEL_KIND == 'b' else 'model_a.npz'


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
    self.model = ModelB() if MODEL_KIND == 'b' else Model()
    self.logger.info(f"Using model {MODEL_KIND.upper()} with weights at {MODEL_PATH}.")
    self.epsilon = 0.2  # exploration rate; tuned later via PLAN.md's hyperparameter grid search

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

    if self.train and np.random.random() < self.epsilon:
        action = np.random.choice(ACTIONS)
        self.logger.debug(f"Exploring: chose random action {action}.")
        return action

    action = ACTIONS[int(np.argmax(q_values))]
    self.logger.debug(f"Exploiting: chose action {action} from Q-values {q_values}.")
    return action
