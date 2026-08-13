"""Dihedral symmetry of the board, for data augmentation.

The rules of the game are invariant under the eight symmetries of the square
(four rotations, optionally mirrored). So every transition we observe is
really eight transitions: if (s, a, r, s') is valid then so is
(Ts, Ta, r, Ts') for every T. That multiplies the effective data by up to
eight at no simulation cost and, for a linear model, forces the four
directional weight blocks to agree with each other instead of each learning
its own quirks from whatever the agent happened to experience.

Two levels:

* `transform_features` / `transform_action` work on the extracted vector and
  are pure index permutations. This is what training uses.
* `transform_state` transforms the raw game_state. It is slow and exists only
  to prove the fast path right: the tests assert
      phi(transform_state(s, t)) == transform_features(phi(s), t)
  on real boards. Without that check the fast path is a very plausible-looking
  way to silently corrupt every training sample.

Transforms are indexed 0..7 as t = 4 * flip + rot: mirror first, then `rot`
quarter turns.
"""

import numpy as np

from .bfs import ACTIONS, DIRECTIONS
from .features import DIRECTIONAL_OFFSETS, FEATURE_DIM

N_TRANSFORMS = 8
IDENTITY = 0


def _decompose(t):
    return divmod(t, 4)   # (flip, rot)


def map_point(x, y, n, t):
    """Image of board coordinate (x, y) under transform t on an n x n board."""
    flip, rot = _decompose(t)
    if flip:
        x = n - 1 - x
    for _ in range(rot):
        x, y = n - 1 - y, x
    return x, y


def map_direction_vector(dx, dy, t):
    """Image of a direction vector under transform t (the linear part only)."""
    flip, rot = _decompose(t)
    if flip:
        dx = -dx
    for _ in range(rot):
        dx, dy = -dy, dx
    return dx, dy


def _build_direction_permutation():
    """perm[t][d]: the direction that direction d becomes under t."""
    lookup = {v: i for i, v in enumerate(DIRECTIONS)}
    perm = np.zeros((N_TRANSFORMS, 4), dtype=np.intp)
    for t in range(N_TRANSFORMS):
        for d, (dx, dy) in enumerate(DIRECTIONS):
            perm[t, d] = lookup[map_direction_vector(dx, dy, t)]
    return perm


DIR_PERM = _build_direction_permutation()


def _build_action_permutation():
    """Moves permute with the directions; BOMB and WAIT are fixed points.

    Built as identity-then-overwrite rather than by appending a hard-coded
    [4, 5], so swapping BOMB and WAIT in the contract needs no edit here.
    """
    perm = np.tile(np.arange(len(ACTIONS), dtype=np.intp), (N_TRANSFORMS, 1))
    perm[:, :4] = DIR_PERM
    return perm


ACTION_PERM = _build_action_permutation()


def _build_feature_permutation():
    """perm[t]: old feature index -> new feature index."""
    perm = np.tile(np.arange(FEATURE_DIM, dtype=np.intp), (N_TRANSFORMS, 1))
    for t in range(N_TRANSFORMS):
        for base in DIRECTIONAL_OFFSETS:
            for d in range(4):
                perm[t, base + d] = base + DIR_PERM[t, d]
    return perm


FEATURE_PERM = _build_feature_permutation()

#: Needed to map an action chosen in transformed space back to the real board.
INV_ACTION_PERM = np.argsort(ACTION_PERM, axis=1)
INV_FEATURE_PERM = np.argsort(FEATURE_PERM, axis=1)


def transform_features(phi, t):
    """Apply transform t to a feature vector or a batch of them.

    Accepts shape (FEATURE_DIM,) or (n, FEATURE_DIM); returns a new array.
    """
    if phi is None:
        return None
    phi = np.asarray(phi)
    out = np.empty_like(phi)
    out[..., FEATURE_PERM[t]] = phi
    return out


def transform_action(action, t):
    """Apply transform t to an action, given as name or as index."""
    if isinstance(action, str):
        return ACTIONS[ACTION_PERM[t][ACTIONS.index(action)]]
    return int(ACTION_PERM[t][action])


def inverse_transform_action(action, t):
    """Map an action chosen on the transformed board back to the real one."""
    if isinstance(action, str):
        return ACTIONS[INV_ACTION_PERM[t][ACTIONS.index(action)]]
    return int(INV_ACTION_PERM[t][action])


def apply_symmetry(features, action=None, transform_id=0):
    """One transform applied to a feature vector and its action.

    This is the signature agreed in interface_contract.md; it is a thin
    wrapper over transform_features / transform_action. The action keeps
    whichever type it came in as, and stays None if None was passed.

    For a whole batch, `augment_batch` is far faster.
    """
    if not 0 <= transform_id < N_TRANSFORMS:
        raise ValueError(f'transform_id must be in [0, {N_TRANSFORMS - 1}], '
                         f'got {transform_id}')
    moved = transform_features(features, transform_id)
    if action is None:
        return moved, None
    return moved, transform_action(action, transform_id)


def augment(phi, action, include_identity=True):
    """Yield (phi_t, action_t) for every symmetry of one observation.

    Duplicates are not filtered: on a symmetric board a few transforms can
    coincide, but the fixed starting corners and the random crate layout make
    that rare enough that de-duplication costs more than it saves.
    """
    start = 0 if include_identity else 1
    for t in range(start, N_TRANSFORMS):
        yield transform_features(phi, t), transform_action(action, t)


def augment_batch(phis, actions):
    """Vectorised `augment` over arrays. Returns (8n, FEATURE_DIM) and (8n,).

    The eight transforms are laid out in the same block order on every call,
    so augmenting states and next-states separately keeps each pair under the
    same transform. Getting that wrong pairs a state with a rotated successor
    and quietly poisons every TD target.
    """
    phis = np.asarray(phis)
    actions = np.asarray(actions, dtype=np.intp)
    n = len(phis)
    out_phi = np.empty((N_TRANSFORMS * n, phis.shape[1]), dtype=phis.dtype)
    out_act = np.empty(N_TRANSFORMS * n, dtype=np.intp)
    for t in range(N_TRANSFORMS):
        out_phi[t * n:(t + 1) * n, FEATURE_PERM[t]] = phis
        out_act[t * n:(t + 1) * n] = ACTION_PERM[t][actions]
    return out_phi, out_act


def augment_transitions(batch, include_identity=True):
    """Expand a list of transitions into one per symmetry: eight times the data.

    This is the form model.update() wants -- a list of Transition -- so the
    whole augmentation step is `model.update(augment_transitions(batch))`.

    Items are duck-typed through `namedtuple._replace` rather than importing
    Transition, so this module does not depend on which model is in use.
    `next_features` is None on a terminal transition and stays None. `reward`
    and `done` are carried through untouched: a rotated board pays the same
    reward and ends the same way.
    """
    start = 0 if include_identity else 1
    out = []
    for item in batch:
        for t in range(start, N_TRANSFORMS):
            out.append(item._replace(
                features=transform_features(item.features, t),
                action=transform_action(item.action, t),
                next_features=transform_features(item.next_features, t),
            ))
    return out


def _map_array(arr, t):
    n = arr.shape[0]
    out = np.empty_like(arr)
    for x in range(n):
        for y in range(arr.shape[1]):
            nx, ny = map_point(x, y, n, t)
            out[nx, ny] = arr[x, y]
    return out


def transform_state(game_state, t):
    """Geometrically transform a whole game_state dict.

    Only defined for square boards, which is what settings.py gives us.
    Deliberately written the slow, obvious way: this is a test oracle, not
    production code.
    """
    if game_state is None:
        return None
    if t == IDENTITY:
        return game_state

    field = game_state['field']
    n = field.shape[0]
    assert field.shape[0] == field.shape[1], 'transform_state needs a square board'

    def pt(p):
        return map_point(p[0], p[1], n, t)

    name, score, bombs_left, pos = game_state['self']
    out = dict(game_state)
    out['field'] = _map_array(field, t)
    out['explosion_map'] = _map_array(game_state['explosion_map'], t)
    out['self'] = (name, score, bombs_left, pt(pos))
    out['others'] = [(n_, s_, b_, pt(p_)) for (n_, s_, b_, p_) in game_state['others']]
    out['bombs'] = [(pt(p_), timer) for (p_, timer) in game_state['bombs']]
    out['coins'] = [pt(c) for c in game_state['coins']]
    return out
