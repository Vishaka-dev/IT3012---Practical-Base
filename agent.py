# agent.py
import random


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        # Simple heuristic or fallback random sweep
        return random.choice(self.actions_pool)


class SimpleReflexAgent:
    """A memoryless reflex agent: acts purely on the current percept via fixed IF-THEN rules.

    No __init__, no state on self - by design. In a partially observable
    environment this means it can loop forever (e.g. turning in place in a
    dead end) since it can never remember what it already tried.
    """

    def sense_and_act(self, percept: dict) -> str:
        if percept['food_here']:
            return 'suck'
        if percept['wall_ahead']:
            return 'turn_left'
        return 'move_forward'


class ModelBasedAgent:
    """A model-based reflex agent: keeps an internal belief state (facing estimate,
    relative position estimate, visited set) inferred purely from its own past
    actions and percepts - it never reads the environment's real agent_pos/facing.
    """

    # The agent's own model of how 'move_forward'/turning affect position - not read
    # from the environment, just a belief about how the world behaves.
    DIRECTION_ORDER = ['N', 'E', 'S', 'W']
    DIRECTION_DELTAS = {'N': (0, 1), 'E': (1, 0), 'S': (0, -1), 'W': (-1, 0)}

    def __init__(self):
        self.facing_estimate = 'N'
        self.position_estimate = (0, 0)
        self.visited = {(0, 0)}
        self.last_action = None
        self.last_wall_ahead = False

    def _turn(self, facing: str, turn: str) -> str:
        idx = self.DIRECTION_ORDER.index(facing)
        if turn == 'right':
            return self.DIRECTION_ORDER[(idx + 1) % 4]
        return self.DIRECTION_ORDER[(idx - 1) % 4]

    def sense_and_act(self, percept: dict) -> str:
        # (a) Transition model: predict how our last action changed our belief state.
        if self.last_action == 'move_forward' and not self.last_wall_ahead:
            dx, dy = self.DIRECTION_DELTAS[self.facing_estimate]
            px, py = self.position_estimate
            self.position_estimate = (px + dx, py + dy)
        elif self.last_action == 'turn_left':
            self.facing_estimate = self._turn(self.facing_estimate, 'left')
        elif self.last_action == 'turn_right':
            self.facing_estimate = self._turn(self.facing_estimate, 'right')

        # (b) Mark the (now updated) estimated position as visited.
        self.visited.add(self.position_estimate)

        # (c) Sensor model: interpret the raw percept using the updated belief state.
        left_facing = self._turn(self.facing_estimate, 'left')
        dx, dy = self.DIRECTION_DELTAS[left_facing]
        px, py = self.position_estimate
        left_cell = (px + dx, py + dy)
        left_is_visited = left_cell in self.visited

        # (d) Condition-action rules over percept + memory.
        if percept['food_here']:
            action = 'suck'
        elif percept['wall_ahead'] and left_is_visited:
            action = 'turn_right'
        elif percept['wall_ahead']:
            action = 'turn_left'
        else:
            action = 'move_forward'

        # (e) Remember what we did, so next call's transition-model step is correct.
        self.last_action = action
        self.last_wall_ahead = percept['wall_ahead']
        return action