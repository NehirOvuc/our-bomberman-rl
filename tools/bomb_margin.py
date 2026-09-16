#!/usr/bin/env python3
"""Would a trained linear model choose BOMB with no opponent present?

Written as the source for a claim about v12/v12b -- two runs of the same
configuration, one of which scores exactly 0.000 in solo on all ten seeds -- and
kept because it overturned that claim.

For every model, Q is evaluated on one shared set of states drawn from replay
buffers, twice: as recorded, and with the opponent-derived features set to zero.
Zeroing is how `solo` looks to the model -- with no opponents on the board,
features.py leaves every one of those columns at 0.

  BOMB wins   share of states where Q(BOMB) is strictly above every other action
  best margin max over states of Q(BOMB) - max(Q(other actions))

The original analysis scored the first 30,000 rows of v12's buffer and found
v12b's best solo-like margin to be -0.738, i.e. no state where it would bomb.
Those rows are the first curriculum stage and contain no crates at all, so there
was nothing to bomb. `--first` reproduces that number; the default random sample
does not. The rows of a buffer are in training order, so sample, never slice.

    python tools/bomb_margin.py \
        --model v12=../wt-run3/agent_code/taco_kebab_agent/model_a.npz \
        --model v12b=../wt-repeat2/agent_code/taco_kebab_agent/model_a.npz \
        --buffer ../wt-run3/agent_code/taco_kebab_agent/replay_buffer.npz \
        --buffer ../wt-repeat2/agent_code/taco_kebab_agent/replay_buffer.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_code.taco_kebab_agent.bfs import ACTIONS  # noqa: E402
from agent_code.taco_kebab_agent.features import FEATURE_DIM, FEATURE_NAMES  # noqa: E402
from agent_code.taco_kebab_agent.model import Model  # noqa: E402

#: Every column computed from other agents' positions. Matched by name rather
#: than index so a reordering in features.py cannot silently change the set.
OPPONENT_FEATURES = [name for name in FEATURE_NAMES
                     if name.startswith('to_opp_')
                     or name in ('opp_dist', 'bomb_hits_opponent')]


def load_states(buffer_paths, n_states, seed, first=False):
    features = np.concatenate([np.load(p)['features'] for p in buffer_paths])
    if features.shape[1] != FEATURE_DIM:
        sys.exit(f'buffer has {features.shape[1]} features, this checkout has '
                 f'{FEATURE_DIM} -- run the script on the commit the models were trained on')
    if first:
        return features[:n_states]
    # One fixed sample shared by every model, so differences between models
    # cannot come from them being scored on different states.
    rng = np.random.default_rng(seed)
    take = rng.choice(len(features), size=min(n_states, len(features)), replace=False)
    return features[take]


def bomb_margin(model, states):
    q = model.predict_q_batch(states)
    bomb = ACTIONS.index('BOMB')
    others = np.delete(q, bomb, axis=1).max(axis=1)
    margin = q[:, bomb] - others
    return float((margin > 0).mean()), float(margin.max())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--model', action='append', required=True,
                        help='label=path to a model_a.npz; repeat for each model')
    parser.add_argument('--buffer', action='append', required=True,
                        help='replay_buffer.npz to draw states from; repeat to pool several')
    parser.add_argument('--states', type=int, default=30_000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--first', action='store_true',
                        help='take the first --states rows instead of a random sample '
                             '(reproduces the original, crate-free analysis)')
    args = parser.parse_args()

    states = load_states(args.buffer, args.states, args.seed, args.first)
    solo_like = states.copy()
    solo_like[:, [FEATURE_NAMES.index(n) for n in OPPONENT_FEATURES]] = 0.0

    print(f'{len(states)} states, {"first rows" if args.first else f"seed {args.seed}"}, crates in {np.mean(states[:, FEATURE_NAMES.index("bomb_crate_count")] > 0):.1%}, zeroed for solo: {", ".join(OPPONENT_FEATURES)}')
    print(f'{"model":<8} {"regime":<15} {"BOMB wins":>10} {"best margin":>12}')
    for spec in args.model:
        label, path = spec.split('=', 1)
        model = Model()
        model.load(path)
        for regime, x in (('with opponents', states), ('solo-like', solo_like)):
            share, best = bomb_margin(model, x)
            print(f'{label:<8} {regime:<15} {share:>9.2%} {best:>+12.3f}')


if __name__ == '__main__':
    main()
