"""Extra scenarios for training only -- Nehir (PLAN.md section 3).

`training-env` branch only. Never merge to master and never ship: the
tournament runs the course's own settings.py, so an agent that needs these
entries to exist is an agent that does not work. The definitions live here
rather than inline in settings.py so the settings.py diff stays five lines --
small enough that nobody merges it by accident.

The four tasks in the brief (section 4) map onto scenarios like this:

  task 1  no crates or opponents, collect revealed coins
          -> coin-heaven                                  (already in settings)
  task 2  crates, no opponents, bomb them open and survive
          -> crate-easy, crate-mid, loot-crate            (the first two are new)
  task 3  crates, hunt peaceful_agent then coin_collector_agent
          -> loot-crate with those opponents
  task 4  crates, hold your own against rule_based_agent
          -> classic                                      (already in settings)

Only task 2 needed new entries, and the reason is measurable. The brief calls
escaping bombs "a crucial capability", so the axis that matters is not crate
density but how often a bomb can be survived at all. Measuring the share of
free tiles from which dropping a bomb leaves an escape (features.py's
bomb_escape_possible, 17x17, averaged over 5 seeds):

    density   0.00  0.20  0.30  0.40  0.50  0.60  0.75
    escapable  100%   96%   90%   80%   68%   54%   32%

Going straight from coin-heaven to loot-crate is one step from 100% to 32%, so
a failure there has no single obvious cause. 0.40 and 0.60 split it into
roughly even drops -- 100, 80, 54, 32 -- and each stage isolates one thing: at
0.40 a suicide is a decision error, because somewhere to run almost always
exists; by 0.60 it is often the position that is already lost.

Coin counts rise with density on purpose. Task 2 asks the agent to find *all*
hidden coins, so a stage with many crates and few coins would measure
crate-clearing rather than coin-finding.

Every stage is reachable: rule_based_agent playing solo, 5 seeds, scores
50/50, 25/25, 33.4/35, 43.2/50 and 8.6/9, and never kills itself. A stage our
agent cannot clear is therefore our agent's problem, not an impossible board.
"""

#: Merged into settings.SCENARIOS. Only CRATE_DENSITY and COIN_COUNT are read
#: (environment.BombeRLeWorld.build_arena).
TRAINING_SCENARIOS = {
    # Task 2, stage 1: bombing is required, dying to it is a decision error.
    'crate-easy': {
        'CRATE_DENSITY': 0.40,
        'COIN_COUNT': 25,
    },
    # Task 2, stage 2: half the tiles no longer offer an escape.
    'crate-mid': {
        'CRATE_DENSITY': 0.60,
        'COIN_COUNT': 35,
    },
}

#: Suggested order and the gate that should pass before moving on. The gates
#: are Member C's metrics from evaluate.py; the thresholds are starting points,
#: to be replaced once the training curves exist.
CURRICULUM = [
    ('coin-heaven', 'task 1', 'collects most reachable coins before the step limit'),
    ('crate-easy',  'task 2', 'suicide rate < 10%'),
    ('crate-mid',   'task 2', 'suicide rate < 5%'),
    ('loot-crate',  'task 2', 'clears crates and finds coins without dying'),
    ('classic',     'task 4', 'beats rule_based_agent on held-out seeds'),
]

# Board size is the other lever, and it is deliberately not here: COLS and ROWS
# are global rather than per-scenario, so shrinking the board changes every
# scenario at once. It also shifts the feature distribution -- distances are
# normalised by features.MAX_DIST = 20, which a 9x9 board never approaches -- so
# weights trained small do not transfer cleanly to 17x17. If training time
# forces it, change COLS/ROWS on this branch and retrain on 17x17 before any
# number goes in the report.