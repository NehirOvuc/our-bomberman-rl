"""Model B — fitted Q-iteration with an ensemble of regression trees
(interface_contract.md section 6, PLAN.md section 2).

One `ExtraTreesRegressor` per action, Q(s, a) = forest_a(phi(s)), rebuilt
from scratch whenever train.py has accumulated another round's worth of
experience (it calls `refit`, never `update` — see below). PLAN.md section 2
requires Model A and Model B to differ *only* in the function approximator:
same phi(s), same reward table, same replay buffer, same epsilon schedule.
That constraint is what fixes almost every choice in this file — wherever
Model A's ridge fit makes a decision (cold start, what a target means, how
paths are handled), this class has to make the matching decision for a
forest rather than inventing its own policy.

`update(batch)` deliberately raises `NotImplementedError`. Model A's
`update` can fold a new batch into its running sufficient statistics because
a normal equation is additive; a fitted forest has no equivalent partial fit
— the only way to make it reflect new data is to regrow the trees, which is
exactly what `refit` already does. Giving `update` real behaviour would just
be a second name for that same rebuild, minus the guarantee that it is being
handed the full buffer `refit` expects, so it is disabled outright instead
of quietly doing something train.py doesn't ask for.
"""

import os

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from .bfs import ACTIONS
from .features import FEATURE_DIM
from .model import Transition  # same namedtuple as Model A; contract section 4

#: Trees per forest. A starting value: the grid search over this and the two
#: constants below is scheduled for PLAN.md section 3, week 3, once there is
#: a trained baseline worth tuning against.
N_ESTIMATORS = 50

#: Depth and leaf-size caps. Both act as regularisers against a forest that
#: memorises 5000 individual transitions instead of generalising across
#: them, and as a ceiling on how large six saved forests can get -- the
#: submission zip has to carry them.
MAX_DEPTH = 12
MIN_SAMPLES_LEAF = 20

#: Below this many rows, an action's share of the batch would be too small to
#: fit anything but noise -- if that ever happens, predict_q keeps that
#: action's previous forest (or its zero cold start, if there's no previous
#: forest yet) rather than fitting on a handful of points.
#:
#: Measured against the current pipeline (REFIT_EVERY, REPLAY_SIZE, and 8x
#: symmetry augmentation): this floor has not been observed to bind at any
#: refit so far, including the first one -- even BOMB, the rarest action,
#: already clears 200 rows by then. So today it is a guard against a
#: configuration this file does not control (a shorter REFIT_EVERY, weaker or
#: disabled augmentation, or an early curriculum stage reaching refit before
#: enough real transitions accumulate), not an active safeguard under current
#: settings. It should stay -- just not be described as doing work it isn't
#: currently doing.
#:
#: Separately: predict_q's zero cold start is not a neutral placeholder once
#: training is underway. Movement Q-values run strongly negative early
#: (roughly -1.8 to -3.1) and slightly positive on a trained forest
#: (~+0.109 mean, measured over 20,000 real states) -- so 0.0 reads as
#: optimistic early and pessimistic late, and it's a distribution shift
#: rather than a clean flip: 0.0 ranks below all four movement actions in
#: only about 35% of states. That matters anywhere this floor's fallback
#: value gets compared against an already-trained forest's other actions.
MIN_SAMPLES_TO_FIT = 200

#: train.py's `_refit_from_replay` hands `refit` the *entire* replay buffer
#: every REFIT_EVERY rounds -- up to REPLAY_SIZE raw transitions, already
#: multiplied 8x by symmetry augmentation before it reaches here -- and that
#: buffer is sized to hold a whole training run, not a recent window. Left
#: uncapped, the number of rows an action contributes only grows as training
#: goes on, and so would the cost of fitting six forests, without bound, for
#: the rest of the run. MAX_SAMPLES_PER_ACTION keeps every refit's cost
#: roughly constant instead. A starting value -- see benchmark_model_b.py.
MAX_SAMPLES_PER_ACTION = 5000


class ModelB:
    """Q-function approximated by six independent regression forests, one
    per action, rather than one multi-output regressor with the action
    folded in as an input.

    This mirrors Model A's structure -- one weight vector per action there,
    one forest per action here -- rather than being an independent design
    choice: PLAN.md's "differ only in the approximator" constraint means
    every place Model A treats the six actions as six separate regression
    problems, this class has to as well, or the comparison would also be
    testing "shared vs. separate capacity across actions" and not just
    "linear vs. forest".

    An action with no forest yet predicts 0.0. Model A's cold start is a
    zero-initialised beta_a, which also predicts 0.0 for every state; an
    untrained forest reporting some other value (a mean reward, say) would
    make the two models disagree before either has seen any data, which is
    exactly the kind of extra difference the controlled comparison is not
    supposed to have.
    """

    def __init__(self, n_features: int = FEATURE_DIM, gamma: float = 0.99,
                 n_step: int = 1, n_estimators: int = N_ESTIMATORS,
                 max_depth: int = MAX_DEPTH, min_samples_leaf: int = MIN_SAMPLES_LEAF,
                 n_jobs: int = -1, random_state: int | None = 0):
        self.n_features = n_features
        self.gamma = gamma

        # Not read anywhere in this class -- train.py's _refit_from_replay
        # reads self.model.n_step off whichever model it has to decide how
        # many rewards to fold into a target before calling refit(). Carried
        # here purely so that attribute exists on both models alike.
        self.n_step = n_step

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

        # Consulted only inside _fit_forest, while a tree ensemble is being
        # grown -- training-time parallelism, which the "no multiprocessing
        # in the final agent" rule allows. See _fit_forest for how a stored
        # forest is kept from ever using more than one core again.
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.actions = ACTIONS
        # None until an action's first successful fit; see the class
        # docstring for why predict_q treats that the same as Model A's
        # zero-initialised beta_a.
        self._forests = {action: None for action in self.actions}

    @property
    def is_fitted(self) -> bool:
        return any(forest is not None for forest in self._forests.values())

    def _validate_shape(self, features: np.ndarray) -> np.ndarray:
        """Check that `features` has the expected (n_features,) shape.

        Shared by `_as_row` (which then casts and reshapes for a single
        predict() call) and `refit` (which needs the check but not that
        reshape, since it stacks many rows into one (n, n_features) batch).
        Returns the array unchanged -- no cast, no reshape -- so each caller
        can do whatever it needs on top without this helper working against
        it.
        """
        if features.shape != (self.n_features,):
            raise ValueError(
                f"Expected features of shape ({self.n_features},), got {features.shape}"
            )
        return features

    def _as_row(self, features: np.ndarray) -> np.ndarray:
        """Validate a single feature vector and shape it for sklearn's predict().

        A wrongly-shaped array handed to predict() would not raise -- numpy
        broadcasting rules would silently produce *some* answer, wrong in a
        way nothing downstream would notice until the agent's behaviour was
        already strange. Raising here converts that into an error where the
        mistake actually happened.
        """
        features = np.asarray(features, dtype=np.float32)
        features = self._validate_shape(features)
        return features.reshape(1, -1)

    def predict_q(self, features: np.ndarray) -> np.ndarray:
        row = self._as_row(features)
        q_values = np.zeros(len(self.actions), dtype=np.float32)
        for i, action in enumerate(self.actions):
            forest = self._forests[action]
            if forest is not None:
                q_values[i] = forest.predict(row)[0]
        return q_values

    def predict_q_batch(self, features: np.ndarray) -> np.ndarray:
        """Q-values for many states at once: (n, n_features) -> (n, n_actions).

        train.py's _refit_from_replay needs max_a Q(s', a) for every
        transition in the replay buffer before it can build the next round
        of n-step targets. Calling predict_q one state at a time in a Python
        loop over a buffer that can hold hundreds of thousands of rows would
        make that bootstrap step, not the tree-growing itself, the slow part
        of every refit -- one call to forest.predict() per action, over the
        whole batch, avoids that.
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
        """Disabled on purpose -- see the module docstring for why.

        Both ways of honouring this call would be worse than refusing it.
        Fitting only on `batch` would discard every transition from earlier
        rounds and quietly hand back a model trained on the latest round
        alone. Rebuilding from everything this instance has ever seen would
        be correct but would mean regrowing six forests on every call
        instead of only when train.py actually decides to refit. So this
        raises, and `refit(batch)` -- called with whatever batch the caller
        intends to be the whole picture -- is the real entry point;
        interface_contract.md section 4 documents that split.
        """
        raise NotImplementedError(
            "ModelB has no incremental update; call refit(batch) instead "
            "(see interface_contract.md section 4)."
        )

    def _fit_forest(self, action: str, phi: np.ndarray, targets: np.ndarray) -> None:
        forest = ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        forest.fit(phi, targets)
        # Parallel fitting only pays for itself during training. A forest
        # that has been stored must never touch more than one core again --
        # not in this process, and not in a tournament process that later
        # loads it from disk (see load()) -- which is what makes "no
        # multiprocessing in the final agent" hold by construction instead
        # of by remembering to check for it.
        forest.n_jobs = 1
        self._forests[action] = forest

    def refit(self, batch: list[Transition]) -> None:
        """Rebuild every action's forest from scratch on `batch`.

        `batch` is not a small mini-batch here: train.py calls this once
        every REFIT_EVERY rounds with the entire current replay buffer, and
        that buffer only grows as a training run continues. Two limits on
        what actually reaches forest.fit() follow directly from that, and
        both are explained where they're defined: MIN_SAMPLES_TO_FIT (an
        action with too little data this round keeps its previous forest,
        or None, rather than fitting on noise) and MAX_SAMPLES_PER_ACTION
        (an action with more data than that is subsampled down to exactly
        the cap, so refit cost does not grow without bound over a long run).

        The subsample is drawn from a `np.random.default_rng` seeded fresh
        from `self.random_state` at the start of this call -- not reused
        across calls, not shared with the state fed to the forests
        themselves -- so that refitting the same buffer under the same
        random_state always draws the same rows and therefore predicts the
        same way; that reproducibility is what makes it possible to compare
        MAX_SAMPLES_PER_ACTION candidates against each other fairly.

        Every reward in `batch` is taken as the target to fit directly, on
        the same assumption Model A's update() makes explicit: whoever built
        the batch (train.py) has already turned raw per-step rewards into
        full n-step TD targets, so this only performs the regression
        phi(s) -> target and never looks at next_features or done.
        """
        by_action: dict[str, list] = {action: [] for action in self.actions}
        for transition in batch:
            by_action[transition.action].append(transition)

        rng = np.random.default_rng(self.random_state)

        for action, rows in by_action.items():
            if len(rows) < MIN_SAMPLES_TO_FIT:
                continue

            if len(rows) > MAX_SAMPLES_PER_ACTION:
                keep = rng.choice(len(rows), size=MAX_SAMPLES_PER_ACTION, replace=False)
                rows = [rows[i] for i in keep]

            phi = np.stack([
                self._validate_shape(np.asarray(t.features, dtype=np.float32))
                for t in rows
            ])
            targets = np.array([t.reward for t in rows], dtype=np.float64)
            self._fit_forest(action, phi, targets)

    def save(self, path: str) -> None:
        if os.path.isabs(path):
            raise ValueError("ModelB.save requires a relative path")

        # sklearn estimators have no plain-data format of their own, so this
        # uses joblib rather than model.py's np.savez -- interface_contract.md
        # section 6 assigns joblib to Model B for exactly that reason.
        # Imported here rather than at module scope so nothing on act()'s
        # import path pays for it unless a save actually happens.
        import joblib

        joblib.dump(
            {
                'forests': self._forests,
                'n_features': self.n_features,
                'gamma': self.gamma,
                'n_step': self.n_step,
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'min_samples_leaf': self.min_samples_leaf,
                'random_state': self.random_state,
            },
            path,
            compress=3,  # six forests are large; the submission zip has to carry them
        )

    def load(self, path: str) -> None:
        if os.path.isabs(path):
            raise ValueError("ModelB.load requires a relative path")

        import joblib

        data = joblib.load(path)

        self._forests = data['forests']
        self.n_features = int(data['n_features'])
        self.gamma = float(data['gamma'])
        self.n_step = int(data.get('n_step', 1))  # older files predate n_step
        self.n_estimators = int(data['n_estimators'])
        self.max_depth = data['max_depth']
        self.min_samples_leaf = int(data['min_samples_leaf'])
        self.random_state = data['random_state']

        # A forest fit before the n_jobs=1 rule existed -- or restored from
        # a file some other code path wrote -- could still carry n_jobs=-1.
        # Re-pinning it here means a loaded model can never spawn workers in
        # the tournament, regardless of how the file it came from was made.
        for forest in self._forests.values():
            if forest is not None:
                forest.n_jobs = 1
