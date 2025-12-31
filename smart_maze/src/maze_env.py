from typing import Tuple, Dict, Set

ACTIONS = {
    0: (0, -1),  # up
    1: (1, 0),   # right
    2: (0, 1),   # down
    3: (-1, 0),  # left
}

class MazeEnv:
    def __init__(self, cfg: Dict):
        self.width: int = cfg['width']
        self.height: int = cfg['height']
        self.start: Tuple[int, int] = tuple(cfg['start'])
        self.goal: Tuple[int, int] = tuple(cfg['goal'])
        self.walls: Set[Tuple[int, int]] = set(cfg['walls'])
        self.traps: Set[Tuple[int, int]] = set(cfg['traps'])
        self.rewards: Dict[str, float] = cfg['rewards']

        self.n_actions = 4
        self.n_states = self.width * self.height
        self.pos: Tuple[int, int] = self.start

    def state_index(self, pos: Tuple[int, int]) -> int:
        return pos[1] * self.width + pos[0]

    def index_to_state(self, idx: int) -> Tuple[int, int]:
        return (idx % self.width, idx // self.width)

    def in_bounds(self, pos: Tuple[int, int]) -> bool:
        return 0 <= pos[0] < self.width and 0 <= pos[1] < self.height

    def reset(self) -> int:
        self.pos = self.start
        return self.state_index(self.pos)

    def step(self, action: int):
        dx, dy = ACTIONS[action]
        next_pos = (self.pos[0] + dx, self.pos[1] + dy)

        reward = self.rewards['step']
        done = False

        if not self.in_bounds(next_pos) or next_pos in self.walls:
            # hit wall or out-of-bounds
            reward += self.rewards['wall']
            next_pos = self.pos  # stay in place
        else:
            self.pos = next_pos
            if self.pos in self.traps:
                reward += self.rewards['trap']
            if self.pos == self.goal:
                reward += self.rewards['goal']
                done = True

        return self.state_index(self.pos), reward, done

    def render_data(self) -> Dict:
        return {
            'width': self.width,
            'height': self.height,
            'pos': self.pos,
            'start': self.start,
            'goal': self.goal,
            'walls': self.walls,
            'traps': self.traps,
        }
