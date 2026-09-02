"""Model B — fitted Q-iteration with a regression forest (PLAN.md section 2).

PROTOTYPE. `model.py` and `train.py` are Daniel's files under the working
agreement; this is a branch for him to take or reject, the same way as the
`callbacks.py` resume fix and the replay buffer.

Deliberately a new file rather than an edit to `model.py`, so Model A is not
touched at all and the diff Daniel has to review is one addition. `model.py`'s
own docstring already says Model B "will share the same interface in a future
file", so this is where it said it would be.

Only the function approximator changes. Same phi(s), same rewards, same
epsilon, same seeds, same harness. That is the whole point: section 6 is
supposed to isolate *linear vs non-linear Q on identical inputs*, and it only
does that if nothing else moves.

Why a forest, in one line: five controlled arms (two reward tables, a
curriculum, a forgetting factor, a replay buffer) failed to make the agent
bomb, and the states where a bomb pays are a conjunction -- crate in the blast
AND an escape route exists AND there is time to reach it. A single hyperplane
approximates that badly; axis-aligned trees approximate it well.

ExtraTreesRegressor rather than RandomForestRegressor: it is the regressor the
original fitted-Q-iteration paper used (Ernst, Geurts & Wehenkel, JMLR 2005),
it fits ~4x faster here for the same accuracy class, and its randomised splits
average away more variance -- which matters because our regression targets
contain our own noisy Q estimates.

Measured on this machine, 35k rows x 33 features (one action's share of a
4000-round curriculum), n_estimators=50, max_depth=12, min_samples_leaf=20:

    fit           0.49 s per action  ->  ~3 s per refit
    bootstrap     0.64 s per action  ->  ~3.8 s per refit
    predict       1.08 ms one state  ->  6.5 ms per step (budget is 500 ms)
    on disk       2.5 MB per action  ->  ~15 MB compressed
"""

import os

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from .features import FEATURE_DIM
from .bfs import ACTIONS
from .model import Transition  # same namedtuple; contract section 4

#: Trees per action. 50 is where the fit cost stops buying accuracy on a run
#: this size; it is a starting value, not a tuned one. PLAN.md assigns the grid
#: search over depth/estimators to Daniel.
N_ESTIMATORS = 50

#: Depth and leaf-size caps. These are the disk-size and overfitting controls,
#: not speed controls -- predict time is dominated by sklearn's per-call
#: overhead and barely moves with depth. Uncapped depth at leaf 5 costs 14.4 MB
#: per action instead of 2.5 MB, which is 86 MB of forest in the submission zip.
MAX_DEPTH = 12
MIN_SAMPLES_LEAF = 20

#: An action needs at least this many transitions before it gets a forest.
#: Below it the split criterion is fitting noise, and an unfitted action
#: keeping its zero Q is the more honest answer.
MIN_SAMPLES_TO_FIT = 200


class ModelB:
    """Forest Q-function, one regressor per action.

    Interface parity with `model.Model`: `predict_q`, `predict_q_batch`,
    `refit`, `save`, `load`, and a `gamma` attribute, so `train.py` on this
    branch drives either model without knowing which it has.

    One deliberate deviation from `interface_contract.md`, flagged rather than
    made silently -- see `update()` below.

    Six independent regressors rather than one regressor with the action as an
    input feature. Per-action is what Model A does, so keeping it makes the
    comparison a change of approximator and nothing else; it is also what FQI
    with a small discrete action set normally does, because it lets each action
    partition the state space differently, which is exactly the freedom BOMB
    needs and never had.
    """

    def __init__(self, n_features: int = FEATURE_DIM, gamma: float = 0.99,
                 n_estimators: int = N_ESTIMATORS, max_depth: int = MAX_DEPTH,
                 min_samples_leaf: int = MIN_SAMPLES_LEAF,
                 n_jobs: int = -1, random_state: int | None = 0,
                 n_step: int = 1):
        self.n_features = n_features
        self.gamma = gamma
        #: Length of the TD window train.py builds targets over. Parity with
        #: Model A, which has carried this since the contract. It is read by
        #: _refit_from_replay, not used here: this class only performs the
        #: regression it is handed.
        self.n_step = n_step
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

        #: Cores used *while fitting*, which is training-time only and so is
        #: allowed. Every forest is switched to n_jobs=1 the moment it is fit
        #: (see `_fit_action`), so a loaded model can never spawn workers
        #: inside the tournament: the "no multiprocessing in the final agent"
        #: rule holds by construction rather than by remembering. It is also
        #: faster that way -- thread dispatch costs more than the trees do on a
        #: single state, 14.7 ms against 1.1 ms.
        self.fit_n_jobs = n_jobs
        self.random_state = random_state

        self.actions = ACTIONS
        #: None until an action has been fit at least once. An action with no
        #: forest predicts 0.0, which is what Model A's zero beta does too.
        self._forests: dict[str, ExtraTreesRegressor | None] = {
            action: None for action in self.actions
        }

    @property
    def is_fitted(self) -> bool:
        return any(forest is not None for forest in self._forests.values())

    def _check_shape(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features)
        if features.shape != (self.n_features,):
            raise ValueError(
                f"Expected features of shape ({self.n_features},), got {features.shape}"
            )
        return features

    def predict_q(self, features: np.ndarray) -> np.ndarray:
        """Q for all six actions in this state. Shape (6,), index i = ACTIONS[i]."""
        phi = self._check_shape(features).astype(np.float32).reshape(1, -1)
        q_values = np.zeros(len(self.actions), dtype=np.float32)
        for i, action in enumerate(self.actions):
            forest = self._forests[action]
            if forest is not None:
                q_values[i] = forest.predict(phi)[0]
        return q_values

    def predict_q_batch(self, features: np.ndarray) -> np.ndarray:
        """Q for many states at once: (n, n_features) -> (n, n_actions).

        The refit needs max_a Q(s', a) for every transition in the buffer, so
        this is the hot path of training, not of play. One vectorised call per
        action instead of one per transition: 0.64 s for 210k states against
        minutes if it were looped.
        """
        phi = np.asarray(features, dtype=np.float32)
        if phi.ndim != 2 or phi.shape[1] != self.n_features:
            raise ValueError(
                f"Expected features of shape (n, {self.n_features}), got {phi.shape}"
            )
        q_values = np.zeros((len(phi), len(self.actions)), dtype=np.float64)
        for i, action in enumerate(self.actions):
            forest = self._forests[action]
            if forest is not None:
                q_values[:, i] = forest.predict(phi)
        return q_values

    def update(self, batch: list[Transition]) -> None:
        """Not supported, on purpose. Use `refit`.

        `interface_contract.md` section 4 gives every model an incremental
        `update(batch)`, and Model A honours it by folding the batch into
        accumulated normal equations. A forest has no such statistics: it is
        rebuilt from a full dataset or not at all.

        The two ways to honour the signature anyway are both worse than
        refusing. Fitting on `batch` alone would silently throw away every
        earlier transition and return a model trained on the last round --
        wrong, with no error to notice. Refitting on everything seen so far
        would be correct but would rebuild six forests on every call, which is
        4000 refits over a curriculum instead of 160.

        So this raises, and PLAN.md section 2 already agrees: Model B is
        specified as "periodically re-fit on the replay buffer", which is
        `refit`. Flagging it rather than deviating quietly -- the contract
        needs one line saying `update` is Model A's entry point and `refit` is
        Model B's.
        """
        raise NotImplementedError(
            "ModelB is fitted from a full replay buffer, not incrementally. "
            "Call refit(batch) instead; see interface_contract.md section 4."
        )

    def _fit_action(self, action: str, phi: np.ndarray, y: np.ndarray) -> None:
        forest = ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.fit_n_jobs,
            random_state=self.random_state,
        )
        forest.fit(phi, y)
        # Inference is single-threaded from here on, in this process and in any
        # process that loads the saved file. See fit_n_jobs above.
        forest.n_jobs = 1
        self._forests[action] = forest

    def refit(self, batch: list[Transition]) -> None:
        """Rebuild every action's forest from scratch on `batch`.

        One step of fitted Q-iteration, and the same contract `Model.refit` has
        on the replay-buffer branch: the caller has already recomputed every
        target against the current Q-function, so this only performs the
        regression phi(s) -> target. It does not look at `next_features` or
        `done`.

        An action with too few transitions keeps whatever forest it had, rather
        than getting a fresh one fit on noise. In practice that is only BOMB,
        and only early -- which is the one action we cannot afford to
        mis-estimate.
        """
        by_action: dict[str, list[Transition]] = {a: [] for a in self.actions}
        for transition in batch:
            by_action[transition.action].append(transition)

        for action, transitions in by_action.items():
            if len(transitions) < MIN_SAMPLES_TO_FIT:
                continue
            phi = np.stack([
                self._check_shape(t.features).astype(np.float32) for t in transitions
            ])
            y = np.array([t.reward for t in transitions], dtype=np.float64)
            self._fit_action(action, phi, y)

    def save(self, path: str) -> None:
        # Relative paths only -- absolute ones break the Docker submission test.
        if os.path.isabs(path):
            raise ValueError("ModelB.save requires a relative path")

        # Imported here, not at module scope: joblib is needed to load the
        # forest at inference time (contract section 6), but this keeps the
        # import next to the one thing that uses it and out of act()'s path.
        import joblib

        joblib.dump(
            {
                'n_features': self.n_features,
                'gamma': self.gamma,
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'min_samples_leaf': self.min_samples_leaf,
                'random_state': self.random_state,
                'n_step': self.n_step,
                'forests': self._forests,
            },
            path,
            # ~15 MB compressed against ~60 MB raw, and the zip has to carry it.
            compress=3,
        )

    def load(self, path: str) -> None:
        if os.path.isabs(path):
            raise ValueError("ModelB.load requires a relative path")

        # joblib unpickles, which is arbitrary code execution on a hostile
        # file. Accepted here and only here: the only thing this ever loads is
        # the forest we trained ourselves and shipped inside our own agent
        # directory. Contract section 6 fixes joblib as Model B's format
        # because sklearn estimators have no sane plain-data encoding. Never
        # point this at a path that came from outside the repository.
        import joblib

        data = joblib.load(path)

        self.n_features = int(data['n_features'])
        self.gamma = float(data['gamma'])
        self.n_estimators = int(data['n_estimators'])
        self.max_depth = data['max_depth']
        self.min_samples_leaf = int(data['min_samples_leaf'])
        self.random_state = data['random_state']
        # Files written before n_step existed mean the one-step target.
        self.n_step = int(data.get('n_step', 1))
        self._forests = data['forests']

        # Belt and braces: a file written before _fit_action pinned n_jobs, or
        # by a future edit that forgets to, must still not spawn workers in the
        # tournament.
        for forest in self._forests.values():
            if forest is not None:
                forest.n_jobs = 1
