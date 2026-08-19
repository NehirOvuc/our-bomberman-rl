"""Cross-check the danger model against the simulator by playing real games.

test_bfs.py asserts the bomb-timing offsets derived by reading environment.py.
That is circular: a misreading would be encoded in those tests and pass. Here
the simulator is the oracle instead.

Two checks, and a third that proves the first two can fail. Without it a green
run means nothing: a duration or blast-range error produces zero unpredicted
deaths, so the death check alone would stay silent through both.
"""

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_code.taco_kebab_agent import bfs  # noqa: E402
from agent_code.taco_kebab_agent.bfs import HORIZON, danger_map  # noqa: E402

ROUNDS = 3
MAX_STEPS = 150


def _world_args(seed):
    return Namespace(no_gui=True, fps=15, turn_based=False, update_interval=0.1,
                     save_replay=False, replay=None, make_video=False,
                     continue_without_training=True, log_dir='logs',
                     save_stats=False, match_name='danger', seed=seed,
                     silence_errors=False, scenario='classic',
                     single_process=True)


class Report:
    def __init__(self):
        self.deaths = 0
        self.unpredicted_deaths = 0
        self.bombs = 0
        self.wrong_bombs = 0
        self.detail = []

    def note(self, msg):
        if len(self.detail) < 5:
            self.detail.append(msg)


class _Bomb:
    """What danger_map said one bomb would do, against what it did."""

    def __init__(self, pos, start, predicted):
        self.pos = pos
        self.start = start
        self.predicted = predicted
        self.observed = set()
        self.last = max((k for k, _ in predicted), default=0)

    def saw(self, read, coords):
        self.observed |= {(read - self.start, tuple(c)) for c in coords}

    def done(self, read):
        return read - self.start > HORIZON

    # A round ends the moment the last agent dies, which is very often the step
    # a bomb goes off. Judging that bomb would report the rest of its blast as
    # a phantom -- an artefact of the round ending, not a bug.
    def judgeable(self, reads):
        return self.start + self.last < reads


def _predict_one(danger_fn, field, pos, timer):
    lethal = danger_fn(field, [(pos, timer)], None)
    ks, xs, ys = np.nonzero(lethal)
    return {(int(k), (int(x), int(y))) for k, x, y in zip(ks, xs, ys)}


def _judge(bomb, report):
    report.bombs += 1
    missed = bomb.observed - bomb.predicted
    phantom = bomb.predicted - bomb.observed
    if missed or phantom:
        report.wrong_bombs += 1
        k, tile = sorted(missed or phantom)[0]
        report.note(f'bomb {bomb.pos} seen at step {bomb.start}: '
                    f'{len(missed)} lethal tile-steps missed, {len(phantom)} '
                    f'predicted but never lethal (first at offset {k}, {tile})')


def play(danger_fn=danger_map, rounds=ROUNDS, seed=0):
    """Play headless rounds, judging every prediction as it comes due."""
    from environment import BombeRLeWorld

    world = BombeRLeWorld(_world_args(seed), [('random_agent', False)] * 4)
    report = Report()
    try:
        for _ in range(rounds):
            world.new_round()
            world.user_input = None
            watching, read = {}, 0

            while world.running and world.active_agents and read < MAX_STEPS:
                state = world.get_state_for_agent(world.active_agents[0])
                field = state['field']
                lethal_now = danger_fn(field, state['bombs'],
                                       state['explosion_map'])[0]

                for pos, timer in state['bombs']:
                    if pos not in watching:
                        watching[pos] = _Bomb(
                            pos, read, _predict_one(danger_fn, field, pos, timer))

                alive = {a.name for a in world.active_agents}
                world.do_step()
                survived = {a.name for a in world.active_agents}

                # evaluate_explosions kills exactly on is_dangerous()
                # explosions, and blast_coords[0] is the bomb that made them.
                for exp in world.explosions:
                    origin = tuple(exp.blast_coords[0])
                    if exp.is_dangerous() and origin in watching:
                        watching[origin].saw(read, exp.blast_coords)

                for pos in [p for p, b in watching.items() if b.done(read)]:
                    _judge(watching.pop(pos), report)

                # lethal_now means "ending this step here is fatal", so the
                # tile to look up is where the agent ended, not where it began.
                for a in world.agents:
                    if a.name not in alive:
                        continue
                    if a.name not in survived:
                        report.deaths += 1
                        if not lethal_now[a.x, a.y]:
                            report.unpredicted_deaths += 1
                            report.note(f'{a.name} died at {(a.x, a.y)} on step '
                                        f'{read}, predicted safe')

                read += 1

            for bomb in watching.values():
                if bomb.judgeable(read):
                    _judge(bomb, report)
    finally:
        world.end()
    return report


@pytest.fixture(scope='module')
def real():
    return play()


def test_the_games_produced_enough_to_judge(real):
    """Guards the two tests below: neither means anything on an empty sample."""
    assert real.deaths > 0
    assert real.bombs > 0


def test_no_agent_dies_where_the_model_predicts_safety(real):
    """The failure that matters. Every escape decision rests on this."""
    assert real.unpredicted_deaths == 0, real.detail


def test_each_bomb_is_lethal_exactly_where_and_when_predicted(real):
    """Validates the offsets themselves, which the death check never reaches."""
    assert real.wrong_bombs == 0, real.detail


@pytest.mark.parametrize('shift', [1, -1])
def test_a_one_step_timing_error_would_be_caught(shift):
    """A check that cannot fail proves nothing, so make it fail on purpose."""
    def shifted(field, bombs, explosion_map, extra_bomb=None):
        real_map = danger_map(field, bombs, explosion_map, extra_bomb)
        out = np.zeros_like(real_map)
        for k in range(real_map.shape[0]):
            if 0 <= k - shift < real_map.shape[0]:
                out[k] = real_map[k - shift]
        return out

    broken = play(danger_fn=shifted, rounds=2)
    assert broken.wrong_bombs > 0


def test_a_shorter_blast_would_be_caught(monkeypatch):
    """EXPLOSION_TIMER is the constant the death check is blind to: shortening
    it leaves the tile lethal now, so nobody dies unpredicted -- only the
    per-bomb check notices."""
    monkeypatch.setattr(bfs, 'EXPLOSION_TIMER', 1)
    broken = play(rounds=2)
    assert broken.unpredicted_deaths == 0
    assert broken.wrong_bombs > 0