"""Turn the framework's events into a training reward. Contract §5.

The framework tells us what happened each step as a list of event strings
(events.py). This module turns that list into one float that training uses.
Evaluation never uses it: evaluate.py runs on the framework's real scoring
only, so nothing here may be needed to just play a game.

Eight events in PLAN.md §3 do not exist in the framework and we work them out
ourselves by comparing game states. Four compare the old state with the new
one -- MOVED_TOWARDS_COIN, MOVED_AWAY_FROM_COIN, ESCAPED_BLAST,
STAYED_IN_BLAST. Four are read off the state we acted in alone --
BOMB_NEXT_TO_CRATE, DROPPED_USELESS_BOMB, TRAPPED_SELF, WAITED_IN_DANGER.
That is why reward_from_events takes both states, and why the second group
still fires on the last step of a round, where there is no new state.

Every distance and danger question is answered by bfs.py, the same module
features.py uses, so the rewards and the features can never disagree about
where a coin or a danger is. The bomb events below reuse features.py's own
computation exactly: `can_escape` is the `bomb_escape_possible` feature
(index 32) and `n_crates` is `bomb_crate_count` (index 30).

Decisions (26.08-02.09.2026, details in docs/open-decisions.md):
- A suicide fires both KILLED_SELF and GOT_KILLED. We count only KILLED_SELF,
  so dying costs -5 once, no matter who caused it.
- Danger means: my own tile becomes lethal at some upcoming step, judged by
  bfs.danger_map. Same definition as the in_danger feature.

v1 -> v2, 02.09.2026. v1 was the contract's starting table and nothing else.
Measured over 500 training rounds and 100 evaluation rounds it produced an
agent that never drops a bomb at all: CRATE_DESTROYED (+0.2) against
KILLED_SELF (-5.0) makes one bad bomb cost twenty-five crates, and nothing
paid for the intermediate act the agent has to learn first, which is dropping
a bomb it can escape from. v2 adds the four events PLAN.md already specified.
The v1 numbers are kept as the control arm of the ablation in report §6.

The total is roughly bounded in [-5, +5] per step. "Roughly" because a death
plus a small shaping penalty can give e.g. -5.3; the contract says roughly
and we prefer that over clamping, which would silently distort other sums.
"""

import events as e

from .bfs import (NEVER, bfs_direction, blast_coords, danger_map, find_escape,
                  first_lethal, free_tiles, safe_from)

# Compare the two states.
MOVED_TOWARDS_COIN = 'MOVED_TOWARDS_COIN'
MOVED_AWAY_FROM_COIN = 'MOVED_AWAY_FROM_COIN'
ESCAPED_BLAST = 'ESCAPED_BLAST'
STAYED_IN_BLAST = 'STAYED_IN_BLAST'

# Read off the state we acted in. Added in v2.
BOMB_NEXT_TO_CRATE = 'BOMB_NEXT_TO_CRATE'
DROPPED_USELESS_BOMB = 'DROPPED_USELESS_BOMB'
TRAPPED_SELF = 'TRAPPED_SELF'
WAITED_IN_DANGER = 'WAITED_IN_DANGER'

REWARDS = {
    # The contract's starting table (§5). Events not listed here count as 0.
    e.COIN_COLLECTED: 1.0,
    e.KILLED_OPPONENT: 5.0,
    e.KILLED_SELF: -5.0,
    e.GOT_KILLED: -5.0,
    e.CRATE_DESTROYED: 0.2,
    e.INVALID_ACTION: -0.1,
    e.WAITED: -0.05,
    MOVED_TOWARDS_COIN: 0.1,
    MOVED_AWAY_FROM_COIN: -0.1,
    ESCAPED_BLAST: 0.3,
    STAYED_IN_BLAST: -0.3,

    # v2. The point of these four is to pay for a good bomb *when it is
    # dropped*, instead of four steps later and only if we survive. The full
    # good sequence is now +0.3 (drop) +0.3 (escape) +0.2 per crate, which is
    # a gradient where v1 had none.
    BOMB_NEXT_TO_CRATE: 0.3,
    # Larger in magnitude than the reward it opposes, as the handout asks.
    # Deliberately smaller than the -5 death it usually precedes: this is an
    # early warning, not a second punishment for the same mistake.
    TRAPPED_SELF: -1.0,
    DROPPED_USELESS_BOMB: -0.2,
    WAITED_IN_DANGER: -0.3,
}


def _coin_distance(game_state):
    """Steps to the nearest reachable coin, or -1 if there is none."""
    field = game_state['field']
    _, _, _, here = game_state['self']
    free = free_tiles(field, game_state['bombs'])
    _, dist = bfs_direction(free, here, game_state['coins'])
    return dist


def _in_danger(game_state):
    """True if my tile is going to be lethal at some upcoming step."""
    field = game_state['field']
    _, _, _, here = game_state['self']
    lethal = danger_map(field, game_state['bombs'], game_state['explosion_map'])
    return first_lethal(lethal)[here] < NEVER


def _bomb_events(game_state):
    """Judge the bomb we just dropped, from the state we dropped it in.

    Three outcomes, in priority order. A bomb we cannot escape is TRAPPED_SELF
    whatever else it would have achieved -- dying is not worth a crate. A bomb
    we can escape is worth something if it will actually destroy a crate or
    catch an opponent, and is a wasted turn otherwise.

    The escape test is the same three lines features.py runs for its
    `bomb_escape_possible` feature, so the reward and the feature cannot
    disagree about whether a bomb was survivable.
    """
    field = game_state['field']
    _, _, _, here = game_state['self']
    bombs = game_state['bombs']
    others = [pos for (_, _, _, pos) in game_state['others']]

    coords = blast_coords(field, here)
    destroys_crate = any(field[c] == 1 for c in coords)
    hits_opponent = any(o in coords for o in others)

    # The world as it would be with our bomb on the board.
    v_free = free_tiles(field, bombs, extra_bomb=here)
    v_lethal = danger_map(field, bombs, game_state['explosion_map'], extra_bomb=here)
    can_escape, _ = find_escape(v_free, here, v_lethal, safe_from(v_lethal),
                                blocked_now=set(others))

    if not can_escape:
        return [TRAPPED_SELF]
    if destroys_crate or hits_opponent:
        return [BOMB_NEXT_TO_CRATE]
    return [DROPPED_USELESS_BOMB]


def derive_events(events, old_game_state, new_game_state):
    """The eight PLAN.md events the framework does not provide.

    Empty when `old_game_state` is None, which is the first step of a round.
    When `new_game_state` is None -- the last step, where train.py has no
    successor state to pass -- the four state-comparison events are skipped but
    the four single-state ones still fire. So a bomb that traps us on the step
    we die is still charged for, which is exactly the step it matters most.
    """
    if old_game_state is None:
        return []

    derived = []

    # --- read off the state we acted in --------------------------------
    if e.BOMB_DROPPED in events:
        derived.extend(_bomb_events(old_game_state))

    # Waiting is not itself bad; waiting on a tile that is about to explode
    # is. The plain WAITED penalty is deliberately tiny by comparison.
    if e.WAITED in events and _in_danger(old_game_state):
        derived.append(WAITED_IN_DANGER)

    if new_game_state is None:
        return derived

    # --- compare the two states ----------------------------------------

    # Coin shaping. Skipped on the step a coin was collected: the nearest
    # coin then suddenly becomes a farther one, which would look like moving
    # away right when the agent did the best possible thing.
    if e.COIN_COLLECTED not in events:
        old_dist = _coin_distance(old_game_state)
        new_dist = _coin_distance(new_game_state)
        # Only compare when a coin was reachable in both states. If one side
        # is -1 the two numbers mean different things and comparing is unfair.
        if old_dist >= 0 and new_dist >= 0:
            if new_dist < old_dist:
                derived.append(MOVED_TOWARDS_COIN)
            elif new_dist > old_dist:
                derived.append(MOVED_AWAY_FROM_COIN)

    # Danger shaping: was my tile due to explode, and is it still?
    was = _in_danger(old_game_state)
    now = _in_danger(new_game_state)
    if was and not now:
        derived.append(ESCAPED_BLAST)
    elif was and now:
        derived.append(STAYED_IN_BLAST)

    return derived


def reward_from_events(events, old_game_state, new_game_state) -> float:
    """Total reward for one step. Signature fixed by contract §5.

    Unknown events contribute 0, so a framework event we chose not to
    reward never crashes training.
    """
    events = list(events)

    # A suicide fires both death events. Count the death once: keep
    # KILLED_SELF, drop GOT_KILLED.
    if e.KILLED_SELF in events:
        events = [ev for ev in events if ev != e.GOT_KILLED]

    events += derive_events(events, old_game_state, new_game_state)

    return float(sum(REWARDS.get(ev, 0.0) for ev in events))
