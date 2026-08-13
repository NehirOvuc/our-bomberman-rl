"""Model A — ridge-regularised linear Q-learning (interface_contract.md section 6).

One linear function per action, Q(s, a) = beta_a . phi(s) + bias_a, fitted by
ridge-regularised least squares on n-step TD targets. The bias is folded into
beta_a by augmenting phi(s) with a constant 1.0 feature.

Model B (fitted Q-iteration with a regression forest) will share the same
`predict_q` / `update` / `save` / `load` interface in a future file.
"""

import os
from collections import namedtuple

import numpy as np

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
N_FEATURES = 34

Transition = namedtuple('Transition', [
    'features',       # np.ndarray (34,)
    'action',         # str, one of ACTIONS
    'reward',         # float — output of reward_from_events
    'next_features',  # np.ndarray (34,) or None if terminal
    'done',           # bool
])


class Model:
    """Ridge-regularised linear Q-function, one weight vector per action.

    Assumption: `Transition.reward` is already the full n-step TD target that
    beta_a should be regressed against — it is not a single-step reward that
    this class discounts or bootstraps further. The caller (train.py) is
    responsible for computing the n-step return using `n_step`/`gamma` before
    building the Transition. This class only performs the ridge-regression fit
    of phi(s) -> reward; it does not look at `next_features` or `done`.

    Maintains, per action, the ridge normal-equation sufficient statistics
    A_a = Phi^T Phi + ridge_lambda * I and b_a = Phi^T y, so `update()` can
    fold in a new batch incrementally without keeping the full transition
    history. beta_a = solve(A_a, b_a) is re-solved after each update.
    """

    def __init__(self, ridge_lambda: float = 1.0, n_features: int = N_FEATURES,
                 n_step: int = 1, gamma: float = 0.99):
        self.ridge_lambda = ridge_lambda
        self.n_features = n_features
        self.n_step = n_step
        self.gamma = gamma

        self.actions = ACTIONS
        dim = n_features + 1  # +1 for the bias term

        self._A = {action: ridge_lambda * np.eye(dim, dtype=np.float64) for action in self.actions}
        self._b = {action: np.zeros(dim, dtype=np.float64) for action in self.actions}
        self._beta = {action: np.zeros(dim, dtype=np.float64) for action in self.actions}

    def _augment(self, features: np.ndarray) -> np.ndarray:
        if features.shape != (self.n_features,):
            raise ValueError(
                f"Expected features of shape ({self.n_features},), got {features.shape}"
            )
        return np.concatenate([features.astype(np.float64), [1.0]])

    def predict_q(self, features: np.ndarray) -> np.ndarray:
        phi = self._augment(features)
        q_values = np.array([phi @ self._beta[action] for action in self.actions], dtype=np.float32)
        return q_values

    def update(self, batch: list[Transition]) -> None:
        by_action = {action: [] for action in self.actions}
        for transition in batch:
            by_action[transition.action].append(transition)

        for action, transitions in by_action.items():
            if not transitions:
                continue

            phi = np.stack([self._augment(t.features) for t in transitions])
            y = np.array([t.reward for t in transitions], dtype=np.float64)

            self._A[action] += phi.T @ phi
            self._b[action] += phi.T @ y
            self._beta[action] = np.linalg.solve(self._A[action], self._b[action])

    def save(self, path: str) -> None:
        if os.path.isabs(path):
            raise ValueError("Model.save requires a relative path")

        arrays = {}
        for action in self.actions:
            arrays[f'A_{action}'] = self._A[action]
            arrays[f'b_{action}'] = self._b[action]
            arrays[f'beta_{action}'] = self._beta[action]

        np.savez(
            path,
            ridge_lambda=self.ridge_lambda,
            n_features=self.n_features,
            n_step=self.n_step,
            gamma=self.gamma,
            **arrays,
        )

    def load(self, path: str) -> None:
        if os.path.isabs(path):
            raise ValueError("Model.load requires a relative path")

        data = np.load(path)

        self.ridge_lambda = float(data['ridge_lambda'])
        self.n_features = int(data['n_features'])
        self.n_step = int(data['n_step'])
        self.gamma = float(data['gamma'])

        for action in self.actions:
            self._A[action] = data[f'A_{action}']
            self._b[action] = data[f'b_{action}']
            self._beta[action] = data[f'beta_{action}']
