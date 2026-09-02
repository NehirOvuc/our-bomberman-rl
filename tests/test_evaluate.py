"""Tests for tools/evaluate.py.

Only the pure functions are tested here. Actually playing a match is the
framework's job and is covered by running the tool; what can silently go wrong
is the arithmetic on the statistics file, and that is what these pin down.
"""

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))

import evaluate  # noqa: E402
from evaluate import (append_log, build_parser,  # noqa: E402
                      check_agent_loaded_model, check_model_present, our_key,
                      per_round, summarise)


def _stats(ours, **others):
    return {'by_agent': {'us': ours, **others}}


# --- reading the statistics file ------------------------------------------

def test_missing_counters_read_as_zero():
    # lifetime_statistics is a defaultdict, so a counter that never fired is
    # absent from the JSON rather than present as 0. Reading it with [] would
    # crash on exactly the agents we care about -- ours, which drops no bombs.
    row = per_round(_stats({'score': 10, 'steps': 100, 'rounds': 5}), 'us', 5)
    assert row['bombs'] == 0
    assert row['suicides'] == 0
    assert row['score'] == 2.0


def test_invalid_rate_is_a_share_of_steps_not_a_count_per_round():
    # A long-lived agent takes more steps, so a per-round invalid count would
    # punish it for surviving. The rate must be per step.
    row = per_round(_stats({'invalid': 50, 'steps': 200}), 'us', 10)
    assert row['invalid_rate'] == pytest.approx(0.25)


def test_invalid_rate_of_an_agent_that_never_moved():
    assert per_round(_stats({}), 'us', 10)['invalid_rate'] == 0.0


def test_steps_becomes_steps_survived_per_round():
    row = per_round(_stats({'steps': 800}), 'us', 4)
    assert row['steps_survived'] == 200.0
    assert 'steps' not in row


def test_solo_lineup_has_no_opponent_to_beat():
    assert per_round(_stats({'score': 3}), 'us', 1)['best_opponent'] is None


def test_best_opponent_is_the_strongest_one():
    stats = _stats({'score': 10}, a={'score': 20}, b={'score': 50}, c={'score': 30})
    assert per_round(stats, 'us', 10)['best_opponent'] == 5.0


# --- aggregating across seeds ---------------------------------------------

def test_mean_over_seeds():
    rows = [{'score': 1.0}, {'score': 2.0}, {'score': 3.0}]
    mean, _ = summarise(rows, 'score')
    assert mean == pytest.approx(2.0)


def test_identical_seeds_give_a_zero_width_interval():
    rows = [{'score': 2.0}] * 5
    mean, half_width = summarise(rows, 'score')
    assert mean == pytest.approx(2.0)
    assert half_width == pytest.approx(0.0)


def test_a_single_seed_cannot_have_an_interval():
    # One seed is a number, not a measurement. Say so rather than printing 0.
    _, half_width = summarise([{'score': 2.0}], 'score')
    assert half_width != half_width  # nan


def test_more_spread_gives_a_wider_interval():
    tight = summarise([{'x': 1.0}, {'x': 1.1}, {'x': 0.9}], 'x')[1]
    loose = summarise([{'x': 1.0}, {'x': 5.0}, {'x': -3.0}], 'x')[1]
    assert loose > tight


# --- the experiment log ----------------------------------------------------

def _summary(**over):
    base = {'agent': 'us', 'lineup': 'solo', 'scenario': 'classic',
            'seeds': [900, 901], 'rounds': 100, 'seeds_won': None,
            'seeds_comparable': 0}
    for key in ['score', 'coins', 'kills', 'suicides', 'bombs', 'crates',
                'invalid_rate', 'steps_survived']:
        base[key] = 0.0
        base[f'{key}_ci'] = 0.0
    base.update(over)
    return base


def test_log_writes_one_header_and_appends_after_that(tmp_path):
    path = tmp_path / 'experiment-log.csv'
    append_log(_summary(score=1.5), 'v1', 'first', path=path)
    append_log(_summary(score=2.5), 'v2', 'second', path=path)

    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]['version'] == 'v1' and rows[0]['score_mean'] == '1.5'
    assert rows[1]['version'] == 'v2' and rows[1]['note'] == 'second'
    assert path.read_text().count('score_mean') == 1


def test_log_records_the_seed_range_not_every_seed(tmp_path):
    path = tmp_path / 'log.csv'
    append_log(_summary(seeds=list(range(900, 910))), 'v1', '', path=path)
    assert list(csv.DictReader(open(path)))[0]['seeds'] == '900-909'


def test_log_leaves_seeds_won_blank_when_there_was_nobody_to_beat(tmp_path):
    path = tmp_path / 'log.csv'
    append_log(_summary(), 'v1', '', path=path)
    assert list(csv.DictReader(open(path)))[0]['seeds_won'] == ''


# --- refusing to record a run that measured nothing ------------------------
#
# An absent model file is not an error anywhere downstream: setup() logs and
# carries on with an all-zero model, argmax over six zeros is index 0, and
# ACTIONS[0] is UP. The agent then walks into the top wall for the whole round
# and every metric is a truthful measurement of that. Since *.npz and *.joblib
# are gitignored, this is the default state of a fresh clone.

def _agent_dir(tmp_path, name='taco_kebab_agent'):
    directory = tmp_path / 'agent_code' / name
    (directory / 'logs').mkdir(parents=True)
    return directory


def test_missing_model_file_aborts(tmp_path, monkeypatch):
    _agent_dir(tmp_path)
    monkeypatch.setattr(evaluate, 'ROOT', tmp_path)
    with pytest.raises(SystemExit, match='no trained model'):
        check_model_present('taco_kebab_agent')


@pytest.mark.parametrize('filename', ['model_a.npz', 'model_b.joblib'])
def test_either_model_format_satisfies_the_check(tmp_path, monkeypatch, filename):
    # Model A saves .npz and Model B .joblib (contract section 6); either is a
    # trained model as far as this check is concerned.
    directory = _agent_dir(tmp_path, 'taco_kebab_agent')
    (directory / filename).write_bytes(b'weights')
    monkeypatch.setattr(evaluate, 'ROOT', tmp_path)
    check_model_present('taco_kebab_agent')             # must not raise


def test_an_empty_model_file_does_not_count(tmp_path, monkeypatch):
    # A 0-byte file left by an interrupted save satisfies is_file() and loads
    # as nothing, which is the failure this guard exists to prevent.
    directory = _agent_dir(tmp_path, 'taco_kebab_agent')
    (directory / 'model_a.npz').write_bytes(b'')
    monkeypatch.setattr(evaluate, 'ROOT', tmp_path)
    with pytest.raises(SystemExit, match='no trained model'):
        check_model_present('taco_kebab_agent')


def test_unknown_agent_directory_aborts(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, 'ROOT', tmp_path)
    with pytest.raises(SystemExit, match='no such agent directory'):
        check_model_present('not_an_agent')


def test_fallback_to_an_untrained_model_aborts(tmp_path, monkeypatch):
    # The file check cannot catch a model that exists but is not the one the
    # agent looks for. The agent says so in its own log, so we read it back.
    directory = _agent_dir(tmp_path)
    (directory / 'logs' / 'taco_kebab_agent.log').write_text(
        'INFO: setting up\nERROR: No trained model found at model_b.joblib\n')
    monkeypatch.setattr(evaluate, 'ROOT', tmp_path)
    with pytest.raises(SystemExit, match='played untrained'):
        check_agent_loaded_model('taco_kebab_agent', 'taco_kebab_agent')


def test_a_clean_log_passes(tmp_path, monkeypatch):
    directory = _agent_dir(tmp_path)
    (directory / 'logs' / 'taco_kebab_agent.log').write_text(
        'INFO: Loading model from model_a.npz.\n')
    monkeypatch.setattr(evaluate, 'ROOT', tmp_path)
    check_agent_loaded_model('taco_kebab_agent', 'taco_kebab_agent')  # no raise


def test_a_stale_log_from_another_line_up_is_ignored(tmp_path, monkeypatch):
    """agents.py truncates only the log named after *this* run's agent.

    A mirror run leaves taco_kebab_agent_0.log .. _3.log behind; a later
    tournament run rewrites only taco_kebab_agent.log. Globbing the directory
    would read the stale files and abort a perfectly good evaluation for ever,
    because *.log is gitignored and nothing cleans them up.
    """
    directory = _agent_dir(tmp_path, 'taco_kebab_agent')
    (directory / 'logs' / 'taco_kebab_agent_0.log').write_text(
        'ERROR: No trained model found at model_a.npz\n')     # stale
    (directory / 'logs' / 'taco_kebab_agent.log').write_text(
        'INFO: Loading model from model_a.npz.\n')            # this run
    monkeypatch.setattr(evaluate, 'ROOT', tmp_path)
    check_agent_loaded_model('taco_kebab_agent', 'taco_kebab_agent')  # no raise


def test_a_missing_log_aborts_rather_than_passing(tmp_path, monkeypatch):
    # Absence of evidence is not evidence the model loaded.
    _agent_dir(tmp_path, 'taco_kebab_agent')
    monkeypatch.setattr(evaluate, 'ROOT', tmp_path)
    with pytest.raises(SystemExit, match='cannot read'):
        check_agent_loaded_model('taco_kebab_agent', 'taco_kebab_agent')


# --- the mirror line-up renames our agent ----------------------------------

def test_duplicate_agent_directories_are_resolved():
    # setup_agents renames duplicates to name_0 / name_1 (environment.py), which
    # is what --lineup mirror does when --mirror-agent is our own agent. A plain
    # lookup would raise KeyError on the one line-up designed to use it.
    stats = {'by_agent': {'us_0': {'score': 8, 'steps': 100},
                          'us_1': {'score': 4, 'steps': 100},
                          'us_2': {'score': 2, 'steps': 100}}}
    assert our_key(stats, 'us') == 'us_0'
    row = per_round(stats, 'us', 4)
    assert row['score'] == 2.0
    # ...and our own renamed row must not be counted as an opponent.
    assert row['best_opponent'] == 1.0


def test_unrenamed_agent_still_resolves():
    assert our_key({'by_agent': {'us': {}}}, 'us') == 'us'


def test_a_genuinely_absent_agent_still_raises():
    with pytest.raises(KeyError):
        our_key({'by_agent': {'someone_else': {}}}, 'us')


# --- scenario names come from settings, not from a literal list ------------

def test_scenario_choices_come_from_settings(monkeypatch):
    """Pins that the parser *reads* settings, rather than agreeing with it.

    Two earlier versions of this test were useless. The first compared
    sorted(settings.SCENARIOS) with itself. The second read the choices off
    the parser but only compared them to settings -- and on a master checkout
    the old hardcoded literal happens to list exactly the four scenarios
    settings defines, so reverting the fix still passed.

    The only thing that separates "read from settings" from "a literal that
    matches settings" is a scenario settings has and the literal cannot: the
    training-env branch adds crate-easy and crate-mid the same way.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import settings

    monkeypatch.setitem(settings.SCENARIOS, 'crate-invented', {})
    action = next(a for a in build_parser()._actions if a.dest == 'scenario')

    assert 'crate-invented' in action.choices, (
        'the parser is not reading settings.SCENARIOS')
    assert sorted(action.choices) == sorted(settings.SCENARIOS)
    # argparse never validates a default against choices, so pin it here.
    assert action.default in action.choices
