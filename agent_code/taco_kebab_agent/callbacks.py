"""Callback interface for taco_kebab_agent (interface_contract.md, 'Reference' section).

Owns state -> action only: state_to_features -> Model.predict_q -> pick action.
No training logic lives here -- game_events_occurred / end_of_round /
Model.update live in train.py.
"""

import os

import numpy as np

from .model import ACTIONS, Model

from .features import state_to_features

#: Relative path per interface_contract.md section 6 -- absolute paths break
#: the Docker submission test. Bare filename because SequentialAgentBackend
#: (agents.py) chdirs into this agent's own directory before every callback,
#: so this is already relative to agent_code/taco_kebab_agent/.
MODEL_PATH = 'model_a.npz'


def setup(self):
    """
    Set up the model for this agent. Called once, before act() is ever called.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.model = Model()
    self.epsilon = 0.2  # exploration rate; tuned later via PLAN.md's hyperparameter grid search

    if self.train:
        self.logger.info("Training mode: starting from a fresh, untrained model.")
        return

    if os.path.isfile(MODEL_PATH):
        self.logger.info(f"Loading trained model from {MODEL_PATH}.")
        self.model.load(MODEL_PATH)
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
