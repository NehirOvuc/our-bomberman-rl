#!/usr/bin/env python3
"""Two-level multi-seed comparison for Model B experiments.

Level 1 (training): for each training seed, train a fresh Model B checkpoint
via a `main.py play --train 1` subprocess, with a per-experiment-arm set of
TACO_* environment overrides layered on top of a common base.

Level 2 (evaluation): evaluate each resulting checkpoint by calling directly
into tools/evaluate.py's own `evaluate()` -- the function whose return value
flows into its `report()` and `append_log()` -- rather than shelling out to
`python tools/evaluate.py` and parsing printed text. Importing it means the
mean score for each (arm, seed) pair is a plain float in memory.

Two hardcoded experiments:

    nstep:       n1 (TACO_NSTEP=1) vs n5 (TACO_NSTEP=5)
    cold-start:  baseline (TACO_NSTEP=1) vs
                 floor1 (TACO_NSTEP=1, TACO_MIN_SAMPLES_TO_FIT=1)

For every seed and every arm of the chosen experiment, this trains one
checkpoint and evaluates it, then prints one comparison table across all
seeds per arm -- mean-of-means plus the min/max range, so the spread across
training seeds is visible rather than collapsed into a single number.

Usage:

    python tools/multiseed_compare.py --experiment nstep
    python tools/multiseed_compare.py --experiment cold-start --seeds 1 2 3
    python tools/multiseed_compare.py --experiment nstep --smoke
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

#: Repository root, same derivation as tools/evaluate.py's own ROOT.
ROOT = Path(__file__).resolve().parent.parent

#: tools/ itself, so `import evaluate` finds tools/evaluate.py without
#: turning tools/ into a package (evaluate.py is a plain module, not part of
#: agent_code's package tree).
sys.path.insert(0, str(Path(__file__).resolve().parent))
#: The repo root, so `agent_code.taco_kebab_agent...` resolves as a package
#: -- same pattern tools/benchmark_model_b.py already uses.
sys.path.insert(0, str(ROOT))

import evaluate  # noqa: E402
#: Read from train.py itself rather than copied as a literal: REFIT_EVERY is
#: what actually decides how many training rounds it takes before Model B
#: ever refits and saves a checkpoint. The smoke default below has to stay
#: strictly above whatever this value is *today*, not above 25 forever --
#: importing it means a future change to REFIT_EVERY keeps the smoke default
#: correct automatically instead of silently breaking it again.
from agent_code.taco_kebab_agent.train import REFIT_EVERY  # noqa: E402

#: Two hardcoded experiment definitions. Each arm's dict is layered on top of
#: the common training env (TACO_MODEL, TACO_FRESH, TACO_MODEL_PATH) inside
#: train_checkpoint -- these only carry what varies between the two arms.
EXPERIMENTS = {
    'nstep': {
        'n1': {'TACO_NSTEP': '1'},
        'n5': {'TACO_NSTEP': '5'},
    },
    'cold-start': {
        'baseline': {'TACO_NSTEP': '1'},
        'floor1': {'TACO_NSTEP': '1', 'TACO_MIN_SAMPLES_TO_FIT': '1'},
    },
}


def checkpoint_name(experiment, arm_name, seed):
    """The TACO_MODEL_PATH filename shared by training and evaluation for one
    (experiment, arm, seed) triple.

    Both sides have to agree on this exact string: training writes it to
    disk, and evaluation reads it back via the same env var name. It is a
    bare filename (no directory component), which is the constraint
    callbacks.py enforces on TACO_MODEL_PATH -- see its comment on why
    (SequentialAgentBackend chdirs into the agent's own directory before
    every callback, so a bare name is already relative to
    agent_code/taco_kebab_agent/, regardless of this script's own cwd).
    """
    return f'{experiment}_{arm_name}_seed{seed}.joblib'


def train_checkpoint(experiment, arm_name, arm_overrides, seed, train_rounds):
    """Level 1: train one fresh Model B checkpoint via a main.py subprocess.

    Follows tools/evaluate.py's run_match() convention exactly: cwd=ROOT,
    capture_output=True, text=True, check the returncode, and raise
    RuntimeError with a truncated stderr tail on failure. The one deliberate
    difference is env=env: this subprocess gets an explicit copy of
    os.environ with the arm's overrides layered on top, and never touches
    (or is touched by) this process's real os.environ. That matters because
    evaluate_checkpoint below has to mutate the real os.environ for its own
    subprocess calls (see its docstring) -- training must stay isolated from
    that regardless of which one runs first.
    """
    env = os.environ.copy()
    env['TACO_MODEL'] = 'b'
    # Every checkpoint in this comparison starts from scratch: without this,
    # a stale model_b.joblib -- or a previous arm's checkpoint accidentally
    # sharing a name -- would get refit on top of already-learned weights
    # instead of the fresh start the comparison needs.
    env['TACO_FRESH'] = '1'
    env['TACO_MODEL_PATH'] = checkpoint_name(experiment, arm_name, seed)
    env.update(arm_overrides)

    command = [
        sys.executable, 'main.py', 'play',
        '--agents', 'taco_kebab_agent', 'rule_based_agent',
        'rule_based_agent', 'rule_based_agent',
        '--train', '1',
        '--no-gui',
        '--n-rounds', str(train_rounds),
        '--seed', str(seed),
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f'training failed (experiment={experiment}, arm={arm_name}, '
            f'seed={seed}, exit {result.returncode}):\n'
            f'{result.stderr[-2000:]}')


def evaluate_checkpoint(experiment, arm_name, seed, eval_lineup, eval_rounds, smoke,
                        keep_checkpoints=False):
    """Level 2: evaluate one trained checkpoint via tools/evaluate.py's own
    evaluate() / report() / append_log(), called directly rather than through
    a `python tools/evaluate.py` subprocess.

    evaluate.py's own subprocess calls (run_match, invoked from inside
    evaluate()) do NOT pass env= -- confirmed by reading tools/evaluate.py --
    so they inherit whatever is sitting in *this* process's real os.environ
    at call time. That is the only way to hand those subprocesses our custom
    TACO_MODEL_PATH checkpoint name without editing evaluate.py, so this
    function mutates the real os.environ for the duration of the call.

    Every TACO_* key's prior value (or absence) is snapshotted first and
    restored in a finally block -- including deleting keys that did not
    exist before -- so a raised exception here can never leak a changed
    environment into the next (arm, seed) iteration.
    """
    previous_taco_env = {key: os.environ[key]
                         for key in os.environ if key.startswith('TACO_')}

    os.environ['TACO_MODEL'] = 'b'
    os.environ['TACO_MODEL_PATH'] = checkpoint_name(experiment, arm_name, seed)
    # Deliberately no TACO_FRESH here: this evaluates the checkpoint that
    # training just produced, it must not be replaced with a fresh model.

    try:
        opponents = evaluate.LINEUPS[eval_lineup]
        summary = evaluate.evaluate(
            'taco_kebab_agent', opponents, eval_lineup, 'classic',
            evaluate.EVAL_SEEDS, eval_rounds,
        )
        if not smoke:
            print(evaluate.report(summary))
            evaluate.append_log(
                summary,
                version=f'{experiment}-{arm_name}',
                note=f'multiseed_compare: {experiment} arm={arm_name} seed={seed}',
            )
        return summary
    finally:
        for key in [k for k in os.environ if k.startswith('TACO_')]:
            if key in previous_taco_env:
                os.environ[key] = previous_taco_env[key]
            else:
                del os.environ[key]
        if not keep_checkpoints:
            checkpoint_path = ROOT / 'agent_code' / 'taco_kebab_agent' / \
                checkpoint_name(experiment, arm_name, seed)
            checkpoint_path.unlink(missing_ok=True)


def print_comparison_table(experiment, results):
    """One table per experiment: every arm's per-seed mean scores, plus the
    mean-of-means and the min/max range across seeds -- so the spread across
    training seeds stays visible instead of collapsing into one number.

    `results` maps arm_name -> list of (seed, mean_score) pairs, in the order
    they were evaluated.
    """
    print(f'\n=== {experiment}: comparison across training seeds ===')
    for arm_name, seed_scores in results.items():
        scores = [score for _, score in seed_scores]
        mean_of_means = sum(scores) / len(scores)
        print(f'\narm={arm_name}')
        for seed, score in seed_scores:
            print(f'  seed {seed}: mean score {score:.3f}')
        print(f'  mean-of-means {mean_of_means:.3f}   '
              f'range [{min(scores):.3f}, {max(scores):.3f}]')


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment', required=True, choices=list(EXPERIMENTS))
    # Defaults for --seeds/--train-rounds/--eval-rounds are left as None here
    # and resolved in main(), because --smoke picks different defaults for
    # whichever of these the user did NOT explicitly pass -- that decision
    # needs to know whether each argument was actually given.
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='training seeds (default: 42 43 44, or a single '
                             'seed under --smoke)')
    parser.add_argument('--train-rounds', type=int, default=None,
                        help='rounds per training run (default: 500, or '
                             'REFIT_EVERY + 1 under --smoke)')
    parser.add_argument('--eval-rounds', type=int, default=None,
                        help='rounds per evaluation seed (default: 50, or 5 '
                             'under --smoke)')
    parser.add_argument('--eval-lineup', default='tournament',
                        choices=list(evaluate.LINEUPS))
    parser.add_argument('--smoke', action='store_true',
                        help='fast sanity run: single seed and small round '
                             'counts unless overridden, and never logs to '
                             'the experiment log')
    parser.add_argument('--keep-checkpoints', action='store_true',
                        help='keep the per-run .joblib checkpoints under '
                             'agent_code/taco_kebab_agent/ (default: delete '
                             'each one once it has been evaluated)')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.smoke:
        if args.seeds is None:
            args.seeds = [42]
        if args.train_rounds is None:
            # Must be strictly greater than REFIT_EVERY: train.py only
            # refits (and only then saves) a checkpoint every REFIT_EVERY
            # rounds, so a --smoke run at or below that threshold never
            # produces a saved model and evaluation fails with "No trained
            # model found" -- verified directly against this codebase's
            # current REFIT_EVERY=25 before landing this default.
            args.train_rounds = REFIT_EVERY + 1
        if args.eval_rounds is None:
            args.eval_rounds = 5
    else:
        if args.seeds is None:
            args.seeds = [42, 43, 44]
        if args.train_rounds is None:
            args.train_rounds = 500
        if args.eval_rounds is None:
            args.eval_rounds = 50

    arms = EXPERIMENTS[args.experiment]
    results = {arm_name: [] for arm_name in arms}

    for seed in args.seeds:
        for arm_name, arm_overrides in arms.items():
            print(f'\n--- experiment={args.experiment} arm={arm_name} seed={seed} ---')

            print(f'training ({args.train_rounds} rounds)...')
            train_checkpoint(args.experiment, arm_name, arm_overrides,
                             seed, args.train_rounds)

            print(f'evaluating ({args.eval_rounds} rounds x '
                  f'{len(evaluate.EVAL_SEEDS)} eval seeds)...')
            summary = evaluate_checkpoint(args.experiment, arm_name, seed,
                                          args.eval_lineup, args.eval_rounds,
                                          args.smoke,
                                          keep_checkpoints=args.keep_checkpoints)
            results[arm_name].append((seed, summary['score']))

    print_comparison_table(args.experiment, results)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
