#!/usr/bin/env python3
"""CPU latency micro-benchmark for feature extraction (PLAN.md section 3, Member A).

The tournament gives 0.5 s per step on CPU with no multiprocessing. Feature
extraction is the share of that budget Member A owns, so it is measured rather
than assumed, and re-measured whenever the feature vector changes.

The reported number is p99, not the mean: one slow step loses the agent its
action and shortens the next step's budget, which inside a blast is fatal.

States come from real games rather than synthetic boards, so the mix of open
board, dense crates and crowded end-game matches what the agent actually sees.

    python tools/benchmark_features.py
    python tools/benchmark_features.py --scenario loot-crate --rounds 5
"""

import argparse
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_code.taco_kebab_agent.features import FEATURE_DIM, state_to_features  # noqa: E402
from agent_code.taco_kebab_agent.symmetry import augment_batch  # noqa: E402
import settings as s  # noqa: E402

BUDGET_MS = s.TIMEOUT * 1000.0


def collect_states(scenario, rounds, max_steps, seed):
    from environment import BombeRLeWorld

    args = Namespace(no_gui=True, fps=15, turn_based=False, update_interval=0.1,
                     save_replay=False, replay=None, make_video=False,
                     continue_without_training=True, log_dir='logs',
                     save_stats=False, match_name='benchmark', seed=seed,
                     silence_errors=False, scenario=scenario, single_process=True)

    states = []
    world = BombeRLeWorld(args, [('rule_based_agent', False)] * 4)
    try:
        for _ in range(rounds):
            world.new_round()
            world.user_input = None
            for _ in range(max_steps):
                if not world.running or not world.active_agents:
                    break
                states.extend(world.get_state_for_agent(a)
                              for a in world.active_agents)
                world.do_step()
    finally:
        world.end()
    return states


def time_calls(fn, states, repeats):
    timings = []
    for _ in range(repeats):
        for state in states:
            t0 = time.perf_counter()
            fn(state)
            timings.append(time.perf_counter() - t0)
    return np.array(timings) * 1000.0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scenario', default='classic', choices=list(s.SCENARIOS))
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--max-steps', type=int, default=200)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    states = collect_states(args.scenario, args.rounds, args.max_steps, args.seed)
    ms = time_calls(state_to_features, states, args.repeats)
    median, p95, p99, worst = np.percentile(ms, [50, 95, 99, 100])

    print(f'scenario {args.scenario}, {len(states)} states x {args.repeats} '
          f'repeats, FEATURE_DIM = {FEATURE_DIM}')
    print(f'\nstate_to_features   (n = {len(ms)})')
    print(f'  mean   {ms.mean():7.3f} ms')
    print(f'  median {median:7.3f} ms')
    print(f'  p95    {p95:7.3f} ms')
    print(f'  p99    {p99:7.3f} ms   <- the number that matters')
    print(f'  max    {worst:7.3f} ms')
    print(f'  budget {BUDGET_MS:7.1f} ms   -> p99 uses {100 * p99 / BUDGET_MS:.2f}%')

    # Training-only, but on the critical path of every fit.
    phis = np.stack([state_to_features(st) for st in states[:512]])
    actions = np.zeros(len(phis), dtype=np.intp)
    t0 = time.perf_counter()
    augment_batch(phis, actions)
    dt = time.perf_counter() - t0
    print(f'\naugment_batch  {len(phis)} -> {8 * len(phis)} samples in '
          f'{dt * 1000:.1f} ms ({dt / len(phis) * 1e6:.1f} us per state)')

    share = p99 / BUDGET_MS
    if share < 0.05:
        print('\nOK: p99 under 5% of the step budget, plenty left for the model.')
    elif share < 0.25:
        print('\nOK but watch it: features alone take over 5% of the budget.')
    else:
        print('\nTOO SLOW: over 25% of the budget. Profile before adding features.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
