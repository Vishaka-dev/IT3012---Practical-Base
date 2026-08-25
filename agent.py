# agent.py
import random
import math
from collections import deque
import heapq


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


class SearchAgent:
    """Uninformed search planner over the fully-observable world model from
    get_percept() ('grid_size', 'walls'). Unlike the reflex agents above, this
    plans a full action sequence up front rather than reacting percept-by-percept.

    Uses the environment's absolute 'Up'/'Down'/'Left'/'Right' actions (not
    'move_forward'/'turn_left') so the search state space is just (x, y) - no
    facing/orientation needed, since each action's effect doesn't depend on it.
    """

    # Matches execute_action()'s Up/Down/Left/Right deltas exactly (Up: y+1, Right: x+1).
    ACTION_DELTAS = {'Up': (0, 1), 'Down': (0, -1), 'Left': (-1, 0), 'Right': (1, 0)}
    STEP_COST = 1  # uniform per-move cost - every action above costs the same

    def __init__(self):
        self.plan = []
        self.active_algo = 'BFS'
        self.unreachable = set()  # food targets a search already failed to reach - skip re-trying them

    def sense_and_act(self, percept: dict) -> str:
        if not self.plan:
            start = tuple(percept['agent_pos'])
            food_list = [tuple(f) for f in percept['all_food'] if tuple(f) not in self.unreachable]

            if not food_list:
                return 'suck'  # nothing left to plan toward - existing no-op action

            goal = min(food_list, key=lambda f: abs(f[0] - start[0]) + abs(f[1] - start[1]))

            search_fn = {'BFS': self.bfs_search, 'DFS': self.dfs_search, 'UCS': self.ucs_search}[self.active_algo]
            path = search_fn(start, goal, percept)

            if path is None:
                # Unreachable - remember it so we don't re-run a full search for it every frame.
                self.unreachable.add(goal)
                return 'suck'
            if not path:
                # Already standing on the goal (start == goal): execute_action must still be
                # called once to trigger pickup, since standing there alone doesn't eat it.
                return 'suck'

            self.plan = path

        return self.plan.pop(0)

    def _successors(self, state: tuple, walls: set, width: int, height: int):
        """Yield (action, next_state) for moves that stay in bounds and avoid walls."""
        x, y = state
        for action, (dx, dy) in self.ACTION_DELTAS.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in walls:
                yield action, (nx, ny)

    def _reconstruct_path(self, parent_action_of, goal: tuple) -> list:
        """Walk parent pointers back from goal to start, then reverse into a forward action list."""
        actions = []
        state = goal
        while True:
            parent, action = parent_action_of(state)
            if parent is None:
                break
            actions.append(action)
            state = parent
        actions.reverse()
        return actions

    def bfs_search(self, start, goal, world_model: dict) -> list:
        """FIFO frontier (deque, popleft) - explores in order of number of moves."""
        start, goal = tuple(start), tuple(goal)
        walls = set(world_model['walls'])
        width, height = world_model['grid_size']
        if start == goal:
            return []

        frontier = deque([start])
        reached = {start: (None, None)}  # state -> (parent_state, action_taken)

        while frontier:
            state = frontier.popleft()
            for action, next_state in self._successors(state, walls, width, height):
                if next_state not in reached:
                    reached[next_state] = (state, action)
                    if next_state == goal:
                        return self._reconstruct_path(lambda s: reached[s], goal)
                    frontier.append(next_state)
        return None  # no path exists

    def dfs_search(self, start, goal, world_model: dict) -> list:
        """LIFO frontier (plain list, pop) - explores depth-first, not shortest-path-optimal."""
        start, goal = tuple(start), tuple(goal)
        walls = set(world_model['walls'])
        width, height = world_model['grid_size']
        if start == goal:
            return []

        frontier = [start]
        reached = {start: (None, None)}

        while frontier:
            state = frontier.pop()
            for action, next_state in self._successors(state, walls, width, height):
                if next_state not in reached:
                    reached[next_state] = (state, action)
                    if next_state == goal:
                        return self._reconstruct_path(lambda s: reached[s], goal)
                    frontier.append(next_state)
        return None

    def ucs_search(self, start, goal, world_model: dict) -> list:
        """Priority queue (heapq) ordered by cumulative cost g(n). Unlike BFS/DFS above,
        the goal test happens when a state is POPPED (not when generated) - required for
        optimality, since a cheaper path to the goal can still be sitting in the frontier
        when the goal is first generated via a costlier one.
        """
        start, goal = tuple(start), tuple(goal)
        walls = set(world_model['walls'])
        width, height = world_model['grid_size']
        if start == goal:
            return []

        counter = 0  # tie-breaker so heapq never has to compare state tuples directly
        frontier = [(0, counter, start)]
        reached = {start: (0, None, None)}  # state -> (cost_so_far, parent_state, action_taken)

        while frontier:
            cost, _, state = heapq.heappop(frontier)
            if state == goal:
                return self._reconstruct_path(lambda s: reached[s][1:], goal)
            if cost > reached[state][0]:
                continue  # stale entry - a cheaper path to this state was already processed
            for action, next_state in self._successors(state, walls, width, height):
                new_cost = cost + self.STEP_COST
                if next_state not in reached or new_cost < reached[next_state][0]:
                    counter += 1
                    reached[next_state] = (new_cost, state, action)
                    heapq.heappush(frontier, (new_cost, counter, next_state))
        return None

    def manhattan_distance(self, pos: tuple, goal: tuple) -> int:
        """|dx| + |dy| - admissible heuristic for 4-directional grid movement."""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def euclidean_distance(self, pos: tuple, goal: tuple) -> float:
        """Straight-line distance - also admissible, but underestimates more loosely than Manhattan here."""
        return math.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)

    def astar_search(self, start_pos, goal_pos, walls, grid_size, heuristic_type='manhattan') -> list:
        """Priority queue (heapq) ordered by f = g + h. Same reached_states-at-pop-time graph-search
        convention as ucs_search, reusing the same _successors() helper for bounds/wall checks -
        the only structural difference is each frontier entry carries its own path_taken directly
        instead of being reconstructed from parent pointers afterward.
        """
        heuristics = {
            'manhattan': self.manhattan_distance,
            'euclidean': self.euclidean_distance,
        }
        heuristic = heuristics[heuristic_type]

        start_pos, goal_pos = tuple(start_pos), tuple(goal_pos)
        walls = set(walls)
        width, height = grid_size
        if start_pos == goal_pos:
            return []

        counter = 0  # tie-breaker so heapq never falls through to comparing path_taken (a list, unorderable)
        g0 = 0
        h0 = heuristic(start_pos, goal_pos)
        frontier = [(g0 + h0, counter, g0, start_pos, [])]  # (f_cost, counter, g_cost, current_pos, path_taken)
        reached_states = set()

        while frontier:
            f_cost, _, g_cost, current_pos, path_taken = heapq.heappop(frontier)

            if current_pos == goal_pos:
                return path_taken
            if current_pos in reached_states:
                continue  # stale entry - this state was already expanded via an earlier, cheaper pop
            reached_states.add(current_pos)

            for action, next_pos in self._successors(current_pos, walls, width, height):
                if next_pos in reached_states:
                    continue
                g_new = g_cost + self.STEP_COST
                h_new = heuristic(next_pos, goal_pos)
                counter += 1
                heapq.heappush(frontier, (g_new + h_new, counter, g_new, next_pos, path_taken + [action]))
        return None