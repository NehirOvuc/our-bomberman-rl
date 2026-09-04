#!/usr/bin/env python3
"""Latency and refit-cost benchmark for Model B (PLAN.md section 3, week 3).

Two measurements, both driven from the real ModelB class rather than assumed:

(a) predict_q() latency, with forests fit at the module's current
    hyperparameters (N_ESTIMATORS, MAX_DEPTH, MIN_SAMPLES_LEAF) -- confirms
    inference stays comfortably inside the tournament's 0.5 s/step CPU
    budget. That budget is shared with feature extraction and action
    selection, so Model B must not be the reason a step times out.

(b) refit() time across the whole replay buffer, for a few candidate
    MAX_SAMPLES_PER_ACTION values -- the number model_b.py's own docstring
    says is a starting value, to be tuned from exactly this benchmark.
    train.py hands refit() the ENTIRE replay buffer every REFIT_EVERY
    rounds, so the synthetic dataset here is sized to match: REPLAY_SIZE
    raw transitions, augmented 8x by symmetry (symmetry.N_TRANSFORMS),
    split evenly across the six actions. A real run is not this uniform
    (BOMB is rarer than the movement actions), but even distribution is the
    worst case every action's pool can reach, which is what the cap has to
    survive regardless of how training actually splits the buffer.

    python agent_code/taco_kebab_agent/benchmark_model_b.py
    python agent_code/taco_kebab_agent/benchmark_model_b.py --caps 1000 5000 20000
    python agent_code/taco_kebab_agent/benchmark_model_b.py --rows-per-action 50000   # smaller, faster pool
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import agent_code.taco_kebab_agent.model_b as model_b_module  # noqa: E402
from agent_code.taco_kebab_agent.bfs import ACTIONS  # noqa: E402
from agent_code.taco_kebab_agent.features import FEATURE_DIM  # noqa: E402
from agent_code.taco_kebab_agent.model import Transition  # noqa: E402
from agent_code.taco_kebab_agent.model_b import (  # noqa: E402
    MAX_DEPTH, MIN_SAMPLES_LEAF, MIN_SAMPLES_TO_FIT, N_ESTIMATORS, ModelB)
from agent_code.taco_kebab_agent.symmetry import N_TRANSFORMS  # noqa: E402
from agent_code.taco_kebab_agent.train import REPLAY_SIZE  # noqa: E402
import settings as s  # noqa: E402

BUDGET_MS = s.TIMEOUT * 1000.0

#: What train.py can actually hand refit(): the full replay buffer, already
#: augmented 8x. Split evenly across the six actions for this synthetic
#: benchmark -- see the module docstring for why uniform is the right stress
#: case even though real training isn't uniform.
DEFAULT_ROWS_PER_ACTION = REPLAY_SIZE * N_TRANSFORMS // len(ACTIONS)


def _synthetic_batch(rng, action, n_rows):
    """`n_rows` Transitions for one action, features and rewards both random.

    The values don't need to mean anything -- this measures fit *cost*, not
    fit *quality*; tests/test_model_b.py already covers quality.
    """
    phi = rng.random((n_rows, FEATURE_DIM)).astype(np.float32)
    y = rng.random(n_rows).astype(np.float64)
    return [Transition(features=phi[i], action=action, reward=float(y[i]),
                       next_features=None, done=True) for i in range(n_rows)]


def benchmark_predict_q(rng, n_calls):
    """Time predict_q() on a model fit at today's real hyperparameters."""
    model = ModelB(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                   min_samples_leaf=MIN_SAMPLES_LEAF, random_state=0)
    batch = []
    for action in ACTIONS:
        # A handful of times MIN_SAMPLES_TO_FIT: enough for every action to
        # actually get a forest, not a benchmark of the untrained zero-Q path.
        batch.extend(_synthetic_batch(rng, action, MIN_SAMPLES_TO_FIT * 5))
    model.refit(batch)
    assert model.is_fitted, "benchmark batch was too small to fit any action"

    states = rng.random((n_calls, FEATURE_DIM)).astype(np.float32)
    timings_ms = np.empty(n_calls)
    for i in range(n_calls):
        t0 = time.perf_counter()
        model.predict_q(states[i])
        timings_ms[i] = time.perf_counter() - t0
    return timings_ms * 1000.0


def benchmark_refit(batch, cap):
    """Time one refit() call (all six actions) under a given sampling cap.

    MAX_SAMPLES_PER_ACTION is deliberately a module constant in model_b.py,
    not a constructor parameter -- it is a starting value the docstring there
    says should be tuned from this benchmark, not something callbacks.py or
    train.py ever needs to vary per instance. So candidate values are tried
    here by monkeypatching the module attribute for the duration of one
    refit() call: refit() reads MAX_SAMPLES_PER_ACTION from its enclosing
    module's namespace at call time (ordinary Python global lookup, not
    bound at def time), so this genuinely exercises each candidate rather
    than approximating it from the outside.
    """
    original_cap = model_b_module.MAX_SAMPLES_PER_ACTION
    model_b_module.MAX_SAMPLES_PER_ACTION = cap
    try:
        model = ModelB(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                       min_samples_leaf=MIN_SAMPLES_LEAF, random_state=0)
        t0 = time.perf_counter()
        model.refit(batch)
        return time.perf_counter() - t0
    finally:
        model_b_module.MAX_SAMPLES_PER_ACTION = original_cap


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--predict-calls', type=int, default=2000,
                   help='number of predict_q() calls to time')
    p.add_argument('--caps', type=int, nargs='+',
                   default=[2000, 5000, 10000, 20000],
                   help='candidate MAX_SAMPLES_PER_ACTION values to compare')
    p.add_argument('--rows-per-action', type=int, default=DEFAULT_ROWS_PER_ACTION,
                   help='synthetic per-action pool size for the refit benchmark')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f'--- predict_q() latency  '
          f'(N_ESTIMATORS={N_ESTIMATORS}, MAX_DEPTH={MAX_DEPTH}, '
          f'MIN_SAMPLES_LEAF={MIN_SAMPLES_LEAF}) ---')
    ms = benchmark_predict_q(rng, args.predict_calls)
    median, p95, p99, worst = np.percentile(ms, [50, 95, 99, 100])
    print(f'  n calls {len(ms)}')
    print(f'  mean   {ms.mean():8.4f} ms')
    print(f'  median {median:8.4f} ms')
    print(f'  p95    {p95:8.4f} ms')
    print(f'  p99    {p99:8.4f} ms')
    print(f'  max    {worst:8.4f} ms')
    print(f'  budget {BUDGET_MS:8.1f} ms  -> p99 uses {100 * p99 / BUDGET_MS:.3f}% '
          f'of the step budget')

    print(f'\n--- refit() time across candidate MAX_SAMPLES_PER_ACTION values ---')
    print(f'  building the synthetic replay pool ({args.rows_per_action} rows/action, '
          f'{len(ACTIONS)} actions) -- this is the slow, one-time part...')
    t0 = time.perf_counter()
    batch = []
    for action in ACTIONS:
        batch.extend(_synthetic_batch(rng, action, args.rows_per_action))
    print(f'  built {len(batch)} transitions in {time.perf_counter() - t0:.1f} s')

    for cap in args.caps:
        dt = benchmark_refit(batch, cap)
        print(f'  cap={cap:<7} refit(all {len(ACTIONS)} actions) = {dt:7.3f} s '
              f'({dt / len(ACTIONS) * 1000:7.1f} ms/action)')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
