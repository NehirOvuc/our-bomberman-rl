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

from evaluate import append_log, per_round, summarise  # noqa: E402


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
