"""Evaluate an agent using the framework's own scoring. PLAN.md section 3, Member C.

This tool never imports rewards.py and never will. The handout says our
auxiliary rewards are not present in official games, so a number produced with
them switched on tells us nothing about tournament performance. Everything
here comes from `main.py --save-stats`, which is the same scoring the
tournament uses.

How a run works: one subprocess per seed, each playing `rounds` rounds, with
training off. The framework writes a JSON file per run; we read our agent's
row out of `by_agent` and divide by the number of rounds to get that seed's
mean. Ten seeds give ten means, and the reported interval is a t-interval over
those ten numbers.

Why seeds and not rounds: `environment.py` writes each round's entry as a sum
over all four agents (`end_round`), so a single run gives our agent's total but
never its round-by-round distribution. A seed is the largest unit we can
actually treat as independent, so a seed is the sampling unit. n = 10 for every
interval below, and the report says so.

Usage:

    python tools/evaluate.py --agent taco_kebab_agent
    python tools/evaluate.py --agent taco_kebab_agent --lineup solo --note "v1 table"
    python tools/evaluate.py --agent taco_kebab_agent --lineup mirror \
                             --mirror-agent taco_kebab_v1
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
from scipy import stats

#: Repository root, derived from this file's location. Only this tool uses it;
#: the agent itself still uses relative paths, because the agent is what gets
#: zipped into the Docker image and absolute paths break there.
ROOT = Path(__file__).resolve().parents[1]

#: Evaluation seeds, deliberately disjoint from the 1-500 range we train on.
#: PLAN.md asks for held-out seeds so we cannot report memorised layouts.
EVAL_SEEDS = list(range(900, 910))

#: Opponent line-ups. The names map onto the handout's four tasks.
LINEUPS = {
    # Handout task 4 and the tournament setup. This is the headline number.
    'tournament': ['rule_based_agent'] * 3,
    # Tasks 1 and 2. With nobody else on the board every bomb and every death
    # is ours, so `bombs` and `suicides` mean exactly what they say -- which is
    # what the reward ablation needs to read.
    'solo': [],
    # Task 3, which we would otherwise skip entirely.
    'task3': ['peaceful_agent', 'coin_collector_agent'],
    # 'mirror' is built at runtime from --mirror-agent.
}

#: Per-agent counters the framework tracks (agents.py, EVENT_STAT_MAP). They
#: are absent from the JSON rather than zero when nothing happened, because
#: lifetime_statistics is a defaultdict -- so always read them with .get().
COUNTERS = ['score', 'coins', 'kills', 'suicides', 'bombs', 'crates',
            'invalid', 'moves', 'steps']

#: Where the experiment log lives. One row per evaluation, appended
#: automatically: a log with gaps in it is worse than no log.
#:
#: Deliberately outside docs/, which is gitignored. The log is shared evidence
#: rather than private working notes -- Nehir and Daniel need to read it, and
#: the handout asks for a public repository containing the whole code base. The
#: report is still kept out of the repo, as the handout also requires.
LOG_PATH = ROOT / 'experiments' / 'experiment-log.csv'

LOG_FIELDS = ['date', 'commit', 'agent', 'version', 'scenario', 'lineup',
              'seeds', 'rounds', 'score_mean', 'score_ci', 'coins', 'kills',
              'suicides', 'bombs', 'crates', 'invalid_rate',
              'steps_survived', 'seeds_won', 'note']


def git_commit():
    """Short hash of the code that produced a result, or '?' outside a repo.

    Suffixed with '-dirty' when the working tree has uncommitted changes.
    Without it a row can name a commit that is not what produced it, which is
    worse than naming nothing: the log exists so a number can be traced back
    to the code behind it, and a hash that is confidently wrong breaks exactly
    that.
    """
    try:
        out = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                             cwd=ROOT, capture_output=True, text=True, timeout=10)
        commit = out.stdout.strip() or '?'
        if commit == '?':
            return commit
        dirty = subprocess.run(['git', 'status', '--porcelain'],
                               cwd=ROOT, capture_output=True, text=True, timeout=10)
        return f'{commit}-dirty' if dirty.stdout.strip() else commit
    except (OSError, subprocess.SubprocessError):
        return '?'


#: Model files an agent may carry: Model A saves .npz, Model B .joblib
#: (interface_contract.md section 6).
MODEL_FILES = ('model_a.npz', 'model_b.joblib')

#: What callbacks.setup() logs when it cannot find a model and falls back to an
#: untrained one. Matched literally; if that message is reworded this check
#: goes quiet, so the two belong together in review.
NO_MODEL_MESSAGE = 'No trained model found'


def check_model_present(agent):
    """Refuse to evaluate an agent that has no trained model on disk.

    Without this the run completes and produces a full, plausible table. An
    absent model file is not an error anywhere: setup() logs and carries on
    with an all-zero model, argmax over six zeros returns index 0, and
    ACTIONS[0] is UP -- so the agent walks into the top wall for four hundred
    steps and every metric is a real measurement of that. The row then lands
    in the experiment log looking exactly like a result.

    This is the default state on a fresh clone rather than an edge case:
    *.npz and *.joblib are both gitignored.
    """
    directory = ROOT / 'agent_code' / agent
    if not directory.is_dir():
        raise SystemExit(f'no such agent directory: {directory.relative_to(ROOT)}')
    if not any((directory / name).is_file() for name in MODEL_FILES):
        raise SystemExit(
            f'{agent} has no trained model ({" or ".join(MODEL_FILES)}) in '
            f'{directory.relative_to(ROOT)}.\n'
            'Refusing to evaluate: an untrained model plays a valid-looking '
            'game (it walks UP for the whole round) and would be logged as a '
            'result. Train first, or pass --no-log if you meant to do this.')


def check_agent_loaded_model(agent):
    """Fail if the agent logged that it fell back to an untrained model.

    The file check above cannot catch a model that exists but is not the one
    the agent looks for -- a Model B run with only `model_a.npz` present, say.
    The agent says so in its own log, so read it back after the first match.
    """
    log_dir = ROOT / 'agent_code' / agent / 'logs'
    for log in log_dir.glob('*.log'):
        try:
            if NO_MODEL_MESSAGE in log.read_text(errors='replace'):
                raise SystemExit(
                    f'{agent} logged "{NO_MODEL_MESSAGE}" during the first '
                    f'match ({log.relative_to(ROOT)}).\n'
                    'It played untrained, so these numbers measure nothing. '
                    'Aborting before a row reaches the experiment log.')
        except OSError:
            # An unreadable log is not itself a reason to abort the run.
            continue


def run_match(agent, opponents, scenario, seed, rounds, stats_path):
    """Play one headless match and return the raw statistics dict.

    Training is off: we simply never pass --train, and main.py then sets
    continue_without_training itself, so rounds run to their natural end
    instead of stopping the moment our agent dies.
    """
    command = [
        sys.executable, 'main.py', 'play',
        '--agents', agent, *opponents,
        '--no-gui',
        '--n-rounds', str(rounds),
        '--seed', str(seed),
        '--scenario', scenario,
        '--save-stats', str(stats_path),
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f'match failed (seed {seed}, exit {result.returncode}):\n'
            f'{result.stderr[-2000:]}')
    with open(stats_path) as handle:
        return json.load(handle)


def our_key(stats, agent):
    """The name our agent is recorded under in `by_agent`.

    `setup_agents` (environment.py) renames duplicates to name_0 / name_1
    whenever the same directory appears more than once, which is precisely
    what the mirror line-up does when --mirror-agent is our own agent. The
    plain name is then absent and a direct lookup raises KeyError.
    """
    by_agent = stats['by_agent']
    if agent in by_agent:
        return agent
    if f'{agent}_0' in by_agent:
        return f'{agent}_0'
    raise KeyError(
        f'{agent} is not in by_agent (found {sorted(by_agent)}); '
        'the framework may have renamed it.')


def per_round(stats, agent, rounds):
    """Our agent's per-round means for one match, plus the opponents' best.

    `by_agent` holds lifetime totals over every round, so dividing by the
    number of rounds is the mean. Missing keys are genuine zeros.
    """
    agent = our_key(stats, agent)
    ours = stats['by_agent'][agent]
    row = {key: ours.get(key, 0) / rounds for key in COUNTERS}

    # Invalid-action rate is a proportion of steps, not a per-round count --
    # a long-lived agent would otherwise look worse than a short-lived one
    # simply for having taken more steps.
    steps = ours.get('steps', 0)
    row['invalid_rate'] = ours.get('invalid', 0) / steps if steps else 0.0
    row['steps_survived'] = row.pop('steps')

    # Best opponent score per round, for the win rate. Empty on the solo
    # line-up, where there is nobody to beat.
    others = [v.get('score', 0) / rounds
              for name, v in stats['by_agent'].items() if name != agent]
    row['best_opponent'] = max(others) if others else None
    return row


def summarise(rows, key):
    """Mean and half-width of a 95% t-interval over the per-seed values.

    n is the number of seeds, not the number of rounds. With n = 10 the
    interval is wide, and it is meant to be: a narrower one would be a claim
    the sampling design does not support.
    """
    values = np.array([r[key] for r in rows], dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, float('nan')
    half_width = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return mean, float(half_width)


def evaluate(agent, opponents, lineup, scenario, seeds, rounds, keep_stats=None):
    """Run one match per seed and aggregate. Returns a summary dict.

    `opponents` is the actual list of agent directories; `lineup` is only the
    name it is recorded under, so that 'mirror' reads as 'mirror' in the log
    rather than as three repetitions of a directory name.
    """
    rows = []

    for seed in seeds:
        if keep_stats:
            keep_stats.mkdir(parents=True, exist_ok=True)
            stats_path = keep_stats / f'{agent}_{lineup}_{scenario}_seed{seed}.json'
            raw = run_match(agent, opponents, scenario, seed, rounds, stats_path)
        else:
            # A temp file we do not keep: the aggregate is what matters, and
            # results/ should not fill up with one file per seed by default.
            #
            # A directory rather than NamedTemporaryFile: that holds the file
            # open exclusively on Windows, so main.py cannot write it and the
            # run dies in environment.py's end() with PermissionError -- after
            # playing every round correctly. Only the default path was
            # affected; --keep-stats writes an ordinary file and always worked.
            work = Path(tempfile.mkdtemp(prefix='evaluate_'))
            try:
                raw = run_match(agent, opponents, scenario, seed, rounds,
                                work / 'stats.json')
            finally:
                shutil.rmtree(work, ignore_errors=True)
        rows.append(per_round(raw, agent, rounds))
        if len(rows) == 1:
            check_agent_loaded_model(agent)
        print(f'  seed {seed}: score {rows[-1]["score"]:.2f}/round, '
              f'{rows[-1]["bombs"]:.1f} bombs, {rows[-1]["suicides"]:.2f} suicides')

    summary = {'agent': agent, 'lineup': lineup, 'scenario': scenario,
               'seeds': list(seeds), 'rounds': rounds, 'n_seeds': len(rows)}
    for key in ['score', 'coins', 'kills', 'suicides', 'bombs', 'crates',
                'moves', 'invalid_rate', 'steps_survived']:
        mean, half_width = summarise(rows, key)
        summary[key] = mean
        summary[f'{key}_ci'] = half_width

    # Win rate over seeds: on how many of them did we out-score the strongest
    # opponent? A per-round win rate is not available, because the framework
    # never writes per-round per-agent scores.
    beatable = [r for r in rows if r['best_opponent'] is not None]
    summary['seeds_won'] = (sum(1 for r in beatable if r['score'] > r['best_opponent'])
                            if beatable else None)
    summary['seeds_comparable'] = len(beatable)
    return summary


def append_log(summary, version, note, path=LOG_PATH):
    """Append one row to the experiment log, creating it with a header if new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    seeds = summary['seeds']
    row = {
        'date': date.today().isoformat(),
        'commit': git_commit(),
        'agent': summary['agent'],
        'version': version,
        'scenario': summary['scenario'],
        'lineup': summary['lineup'],
        'seeds': f'{seeds[0]}-{seeds[-1]}' if len(seeds) > 1 else str(seeds[0]),
        'rounds': summary['rounds'],
        'score_mean': round(summary['score'], 3),
        'score_ci': round(summary['score_ci'], 3),
        'coins': round(summary['coins'], 3),
        'kills': round(summary['kills'], 3),
        'suicides': round(summary['suicides'], 3),
        'bombs': round(summary['bombs'], 2),
        'crates': round(summary['crates'], 2),
        'invalid_rate': round(summary['invalid_rate'], 4),
        'steps_survived': round(summary['steps_survived'], 1),
        'seeds_won': (f'{summary["seeds_won"]}/{summary["seeds_comparable"]}'
                      if summary['seeds_won'] is not None else ''),
        'note': note,
    }
    is_new = not path.exists()
    with open(path, 'a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return row


def report(summary):
    """Human-readable summary. Every figure is a per-round mean over seeds."""
    lines = [
        f'{summary["agent"]} | {summary["lineup"]} | {summary["scenario"]} | '
        f'{summary["rounds"]} rounds x {summary["n_seeds"]} seeds',
        '',
        f'  score          {summary["score"]:7.3f} +- {summary["score_ci"]:.3f}  per round',
        f'  coins          {summary["coins"]:7.3f} +- {summary["coins_ci"]:.3f}',
        f'  kills          {summary["kills"]:7.3f} +- {summary["kills_ci"]:.3f}',
        f'  suicides       {summary["suicides"]:7.3f} +- {summary["suicides_ci"]:.3f}',
        f'  bombs          {summary["bombs"]:7.3f} +- {summary["bombs_ci"]:.3f}',
        f'  crates         {summary["crates"]:7.3f} +- {summary["crates_ci"]:.3f}',
        f'  invalid rate   {summary["invalid_rate"]:7.3f} +- {summary["invalid_rate_ci"]:.3f}  of all steps',
        f'  steps survived {summary["steps_survived"]:7.1f} +- {summary["steps_survived_ci"]:.1f}  per round',
    ]
    if summary['seeds_won'] is not None:
        lines.append(f'  seeds won      {summary["seeds_won"]}/{summary["seeds_comparable"]}'
                     f'  (out-scored the best opponent)')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--agent', default='taco_kebab_agent')
    parser.add_argument('--lineup', default='tournament',
                        choices=[*LINEUPS, 'mirror'])
    parser.add_argument('--mirror-agent',
                        help='agent directory to use as all three opponents '
                             'when --lineup mirror is given')
    parser.add_argument('--scenario', default='classic',
                        choices=['empty', 'coin-heaven', 'loot-crate', 'classic'])
    parser.add_argument('--rounds', type=int, default=100)
    parser.add_argument('--seeds', type=int, nargs='+', default=EVAL_SEEDS)
    parser.add_argument('--version', default='',
                        help='label for the experiment log, e.g. "v1" or "v2"')
    parser.add_argument('--note', default='',
                        help='one sentence on what this run was testing')
    parser.add_argument('--keep-stats', action='store_true',
                        help='keep the per-seed JSON files under results/')
    parser.add_argument('--no-log', action='store_true',
                        help='do not append a row to the experiment log')
    args = parser.parse_args()

    lineup = args.lineup
    if lineup == 'mirror':
        if not args.mirror_agent:
            parser.error('--lineup mirror needs --mirror-agent')
        opponents = [args.mirror_agent] * 3
    else:
        opponents = LINEUPS[lineup]

    if not args.no_log:
        # Only a run that will be recorded has to be trustworthy; --no-log is
        # the escape hatch for deliberately measuring an untrained agent, which
        # is how the performance floor in the report was produced.
        check_model_present(args.agent)

    print(f'evaluating {args.agent} vs {opponents or "nobody"} on {args.scenario}, '
          f'{args.rounds} rounds x {len(args.seeds)} seeds, training off')

    summary = evaluate(args.agent, opponents, lineup, args.scenario,
                       args.seeds, args.rounds,
                       keep_stats=(ROOT / 'results') if args.keep_stats else None)
    print()
    print(report(summary))

    if not args.no_log:
        append_log(summary, args.version, args.note)
        print(f'\nlogged to {LOG_PATH.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
